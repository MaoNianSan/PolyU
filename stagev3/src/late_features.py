from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import config
from .api_clients import HuaweiMaaSClient, MaaSAPIError, append_cache, load_cache, now_utc, remove_surrogate_cache_rows, sha256_text
from .normalization import normalize_text_value

PERSON_WORDS = {"boy", "boys", "girl", "girls", "child", "children", "kid", "kids", "mother", "woman", "lady", "son", "daughter", "sister", "brother", "he", "she", "him", "her", "they", "them", "person", "people"}
OBJECT_WORDS = {"cookie", "cookies", "cooky", "jar", "stool", "chair", "sink", "water", "dish", "dishes", "cup", "plate", "curtain", "curtains", "window", "floor", "counter", "cabinet", "apron", "faucet", "towel"}
PLACE_WORDS = {"kitchen", "room", "house", "home", "inside", "outside", "yard", "garden"}
ACTION_WORDS = {"take", "taking", "reach", "reaching", "steal", "stealing", "fall", "falling", "wash", "washing", "drying", "overflow", "overflowing", "spill", "spilling", "stand", "standing", "sit", "sitting", "climb", "climbing", "look", "looking", "see", "doing", "get", "getting"}
SCORE_COLUMNS = [
    "late_sentence_completeness",
    "late_fragmentation_control",
    "late_repetition_control",
    "late_repair_restart_control",
    "late_filler_vague_control",
    "late_grammatical_stability",
    "late_connected_speech_quality",
    "late_overall_expressive_form_quality",
]

LATE_TRAIN_FILE = "train_late_features.csv"
LATE_EXTERNAL_FILE = "external_late_features.csv"
LATE_MANIFEST_FILE = "late_feature_manifest.json"
LATE_CACHE_FIELDNAMES = ["cache_key", "model", "prompt_version", "text_hash", "scores_json", "created_at", "source"]


def _contains_surrogate_marker(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"api_mode", "source", "embedding_source"}:
                if str(item).strip().lower() in {"local_surrogate", "surrogate"}:
                    return True
            if _contains_surrogate_marker(item):
                return True
    elif isinstance(value, list):
        return any(_contains_surrogate_marker(item) for item in value)
    return False


def mask_content(text: str) -> str:
    """Stagev2-style typed content mask adapted to v3.

    The mask preserves expressive form while removing Cookie Theft scene content.
    Plain alphabetic placeholders are used instead of bracketed placeholders to
    reduce provider safety false positives.
    """
    toks = re.findall(r"[A-Za-z']+|\d+|[^\w\s]", normalize_text_value(text))
    out = []
    for tok in toks:
        low = tok.lower()
        if low in PERSON_WORDS:
            out.append("PERSON")
        elif low in OBJECT_WORDS:
            out.append("OBJECT")
        elif low in PLACE_WORDS:
            out.append("PLACE")
        elif low in ACTION_WORDS:
            out.append("ACTION")
        elif re.fullmatch(r"\d+", tok):
            out.append("NUMBER")
        else:
            out.append(tok)
    masked = " ".join(out)
    masked = re.sub(r"\s+([,.;:!?])", r"\1", masked)
    return re.sub(r"\s+", " ", masked).strip()


def _parse_llm_json(content: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", content, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in LLM response")
    data = json.loads(m.group(0))
    out = {}
    for col in SCORE_COLUMNS:
        raw = data.get(col.replace("late_", ""), data.get(col, None))
        out[col] = float(raw) if raw is not None else np.nan
    return out


def _messages(masked_text: str) -> list[dict[str, str]]:
    system = "You score expressive-form quality of a content-masked connected-speech transcript. Return only JSON."
    user = (
        "Given the content-masked transcript, score each item from 0 to 1: "
        "sentence_completeness, fragmentation_control, repetition_control, repair_restart_control, "
        "filler_vague_control, grammatical_stability, connected_speech_quality, overall_expressive_form_quality.\n"
        f"Transcript: {masked_text}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _strong_safety_mask(text: str) -> str:
    return re.sub(r"[A-Za-z']{5,}", "CONTENT", text)


def _neutral_structure_mask(text: str) -> str:
    return re.sub(r"[A-Za-z']+", "CONTENT", text)


def _cache_key(masked_text: str) -> str:
    return sha256_text(config.LATE_LLM_MODEL + "\n" + config.LATE_LLM_PROMPT_VERSION + "\n" + masked_text)


def _manifest_is_valid(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if manifest.get("historical_feature_outputs_reused") is not False:
        return False
    if manifest.get("regenerated_from_raw_text") is not True:
        return False
    if manifest.get("local_surrogate_allowed") is not False:
        return False
    if manifest.get("feature_load_mode") not in {"extracted_this_run", "existing_feature_csv"}:
        return False
    counts = manifest.get("api_or_cache_mode_counts", {}) or {}
    if any(str(k).lower() == "local_surrogate" and int(v) > 0 for k, v in counts.items()):
        return False
    if _contains_surrogate_marker(manifest):
        return False
    return True


def _sample_ids_match(path: Path, expected_ids: pd.Series) -> bool:
    try:
        df = pd.read_csv(path, usecols=["sample_id"])
    except Exception:
        return False
    return df["sample_id"].astype(str).tolist() == expected_ids.astype(str).tolist()


def _load_existing_features(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path) -> tuple[dict[str, pd.DataFrame], dict] | None:
    train_path = output_dir / LATE_TRAIN_FILE
    test_path = output_dir / LATE_EXTERNAL_FILE
    manifest_path = output_dir / LATE_MANIFEST_FILE
    if not (train_path.exists() and test_path.exists() and _manifest_is_valid(manifest_path)):
        return None
    if not (_sample_ids_match(train_path, train["sample_id"]) and _sample_ids_match(test_path, test["sample_id"])):
        return None
    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)
    if bool(getattr(config, "LATE_ADD_V2_DERIVED_FORM_FEATURES", True)):
        required = {"late_form_quality_inverse", "late_token_count", "late_unique_ratio", "late_punctuation_count"}
        if not required.issubset(set(tr.columns)) or not required.issubset(set(te.columns)):
            return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = dict(manifest)
    manifest["feature_load_mode"] = "existing_feature_csv"
    manifest["feature_extracted_this_run"] = False
    return {"train": tr, "test": te}, manifest


def has_complete_real_features_or_cache(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    allow_existing_features: bool = True,
) -> bool:
    if allow_existing_features and _load_existing_features(train, test, output_dir) is not None:
        return True
    cache_path = config.CACHE_DIR / "huawei_llm_late_feature_cache.csv"
    remove_surrogate_cache_rows(cache_path)
    cache = load_cache(cache_path, "cache_key")
    for df in (train, test):
        for text in df["text"].tolist():
            key = _cache_key(mask_content(text))
            row = cache.get(key)
            if row is None or str(row.get("source", "")).lower() != "api":
                return False
            try:
                scores = json.loads(row.get("scores_json", "{}"))
                if any(column not in scores for column in SCORE_COLUMNS):
                    return False
            except Exception:
                return False
    return True


def _delete_existing_features(output_dir: Path) -> None:
    for name in [LATE_TRAIN_FILE, LATE_EXTERNAL_FILE, LATE_MANIFEST_FILE]:
        (output_dir / name).unlink(missing_ok=True)


def _extract_one(masked_text: str, cache: dict[str, dict[str, Any]], client: HuaweiMaaSClient, new_rows: list[dict[str, Any]]) -> tuple[dict[str, float], str, str]:
    key = _cache_key(masked_text)
    if key in cache:
        row = cache[key]
        try:
            parsed = json.loads(row.get("scores_json", "{}"))
            return {c: float(parsed.get(c, np.nan)) for c in SCORE_COLUMNS}, "cache", ""
        except Exception:
            pass
    if not client.has_key():
        raise RuntimeError(
            "Late feature extraction needs LLM API or a complete real-API cache. "
            "MAAS_API_KEY is missing and cached late features do not cover all required transcripts. "
            "Set MAAS_API_KEY or provide output/cache/huawei_llm_late_feature_cache.csv."
        )
    safety_masked = False
    try:
        _, content = client.chat_json(_messages(masked_text))
    except MaaSAPIError as exc:
        if exc.kind != "content safety / 403":
            raise
        try:
            _, content = client.chat_json(_messages(_strong_safety_mask(masked_text)))
        except MaaSAPIError as masked_exc:
            if masked_exc.kind != "content safety / 403":
                raise
            _, content = client.chat_json(_messages(_neutral_structure_mask(masked_text)))
        safety_masked = True
    scores = _parse_llm_json(content)
    cache_row = {
        "cache_key": key,
        "model": config.LATE_LLM_MODEL,
        "prompt_version": config.LATE_LLM_PROMPT_VERSION,
        "text_hash": sha256_text(masked_text),
        "scores_json": json.dumps(scores),
        "created_at": now_utc(),
        "source": "api",
    }
    new_rows.append(cache_row)
    cache[key] = cache_row
    return scores, "api", "content_safety_masked" if safety_masked else ""


def _extract_df(
    df: pd.DataFrame,
    cache: dict[str, dict[str, Any]],
    client: HuaweiMaaSClient,
    new_rows: list[dict[str, Any]],
    progress_callback: Callable[[dict], None] | None = None,
    counters: dict[str, int] | None = None,
    cache_path: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    modes = []
    rows = []
    for _, row in df.iterrows():
        masked = mask_content(row["text"])
        scores, mode, err = _extract_one(masked, cache, client, new_rows)
        if mode == "api" and cache_path is not None and new_rows:
            # Flush each successful late LLM request so a later provider error does
            # not discard already paid API results.
            append_cache(cache_path, new_rows, LATE_CACHE_FIELDNAMES)
            new_rows.clear()
        if counters is not None and mode == "api":
            counters["completed_requests"] += 1
            counters["new_cache_rows"] = counters.get("new_cache_rows", 0) + 1
            if err == "content_safety_masked":
                counters["safety_masked_requests"] += 1
            if progress_callback:
                progress_callback({"completed_requests": counters["completed_requests"]})
        modes.append(mode)
        item = {"sample_id": row["sample_id"], "masked_text": masked, "late_llm_mode": mode, "late_llm_error": err}
        item.update(scores)
        rows.append(item)
    out = pd.DataFrame(rows)
    out["late_form_risk_score"] = 1 - out[SCORE_COLUMNS].mean(axis=1)
    if bool(getattr(config, "LATE_ADD_V2_DERIVED_FORM_FEATURES", True)):
        # v2-compatible derived form features. The current cache stores 0-1 scores;
        # these derived variables are scale-compatible after StandardScaler and restore
        # the missing token/lexical/punctuation form signals used by v2.
        out["late_form_quality_inverse"] = 1 - out["late_overall_expressive_form_quality"]
        raw_texts = df["text"].fillna("").astype(str).tolist()
        tokenized = [re.findall(r"[A-Za-z']+", t.lower()) for t in raw_texts]
        out["late_token_count"] = [float(max(len(toks), 1)) for toks in tokenized]
        out["late_unique_ratio"] = [float(len(set(toks)) / max(len(toks), 1)) for toks in tokenized]
        out["late_punctuation_count"] = [float(len(re.findall(r"[.!?]", t))) for t in raw_texts]
    return out, modes


def generate_late_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    force_extract: bool = False,
    reuse_features: bool = True,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if force_extract:
        _delete_existing_features(output_dir)
    if reuse_features:
        existing = _load_existing_features(train, test, output_dir)
        if existing is not None:
            if progress_callback:
                total = len(train) + len(test)
                progress_callback({
                    "total_transcripts": total,
                    "cached_transcripts": total,
                    "pending_api_transcripts": 0,
                    "completed_requests": 0,
                })
            return existing
        _delete_existing_features(output_dir)

    cache_path = config.CACHE_DIR / "huawei_llm_late_feature_cache.csv"
    removed_surrogate_rows = remove_surrogate_cache_rows(cache_path)
    cache = load_cache(cache_path, "cache_key")
    client = HuaweiMaaSClient()
    new_rows: list[dict[str, Any]] = []
    masked_all = [mask_content(text) for df in (train, test) for text in df["text"].tolist()]
    cached = sum(1 for text in masked_all if _cache_key(text) in cache)
    if progress_callback:
        progress_callback({
            "total_transcripts": len(masked_all),
            "cached_transcripts": cached,
            "pending_api_transcripts": len(masked_all) - cached,
            "completed_requests": 0,
        })
    counters = {"completed_requests": 0, "safety_masked_requests": 0, "new_cache_rows": 0}
    tr, modes_tr = _extract_df(train, cache, client, new_rows, progress_callback, counters, cache_path)
    te, modes_te = _extract_df(test, cache, client, new_rows, progress_callback, counters, cache_path)
    append_cache(cache_path, new_rows, LATE_CACHE_FIELDNAMES)
    tr.to_csv(output_dir / LATE_TRAIN_FILE, index=False, encoding="utf-8")
    te.to_csv(output_dir / LATE_EXTERNAL_FILE, index=False, encoding="utf-8")
    mode_counts = pd.Series(modes_tr + modes_te).value_counts().to_dict()
    manifest = {
        "feature_family": "late_llm_content_masked_expressive_form",
        "regenerated_from_raw_text": True,
        "historical_feature_outputs_reused": False,
        "api_logic_source": "stage2_equivalent_rewrite",
        "api_or_cache_mode_counts": {str(k): int(v) for k, v in mode_counts.items()},
        "feature_load_mode": "extracted_this_run",
        "feature_extracted_this_run": True,
        "local_surrogate_allowed": False,
        "cache_path": str(cache_path),
        "new_cache_rows": int(counters.get("new_cache_rows", 0)),
        "removed_surrogate_cache_rows": int(removed_surrogate_rows),
        "safety_masked_api_calls": counters["safety_masked_requests"],
        "n_train_rows": int(len(tr)),
        "n_external_rows": int(len(te)),
        "n_late_features": int(len([c for c in tr.columns if c.startswith("late_") and pd.api.types.is_numeric_dtype(tr[c])])),
        "v2_derived_form_features_added": bool(getattr(config, "LATE_ADD_V2_DERIVED_FORM_FEATURES", True)),
    }
    (output_dir / LATE_MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"train": tr, "test": te}, manifest
