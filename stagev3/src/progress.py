from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from . import config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunProgress:
    def __init__(
        self,
        run_mode: str,
        cv_mode: str,
        stability_seeds: list[int],
        selected_model_count: int = 0,
    ) -> None:
        config.ensure_dirs()
        now = _now()
        expected = config.expected_main_rows()
        if run_mode == "all":
            expected += config.expected_main_rows() * len(stability_seeds)
        elif run_mode == "stability":
            expected = config.expected_main_rows() * len(stability_seeds)
        elif run_mode == "selected_after_seed2026":
            expected = selected_model_count * len(stability_seeds)
        self.path = config.FINAL_REPORT_DIR / "run_progress.json"
        self.md_path = config.FINAL_REPORT_DIR / "run_progress.md"
        self.data: dict[str, Any] = {
            "project": config.PROJECT_NAME,
            "run_mode": run_mode,
            "cv_mode": cv_mode,
            "status": "running",
            "current_stage": "initializing",
            "main_seed": config.PRIMARY_SEED,
            "stability_seeds": stability_seeds if run_mode in {"stability", "all", "selected_after_seed2026"} else [],
            "selected_model_count": int(selected_model_count),
            "total_selected_runs": int(selected_model_count * len(stability_seeds)),
            "total_runs": int(selected_model_count * len(stability_seeds)),
            "current_seed": None,
            "current_selected_model_index": 0,
            "feature_status": {"early": "pending", "middle": "pending", "late": "pending"},
            "middle_embedding": {
                "total_windows": 0,
                "cached_windows": 0,
                "pending_api_windows": 0,
                "api_batch_size": config.EMBEDDING_BATCH_SIZE,
                "completed_batches": 0,
            },
            "late_features": {
                "total_transcripts": 0,
                "cached_transcripts": 0,
                "pending_api_transcripts": 0,
                "completed_requests": 0,
            },
            "model_progress": {
                "completed_model_runs": 0,
                "expected_model_runs": expected,
            },
            "started_at": now,
            "updated_at": now,
        }
        self.write()

    def update(self, **changes: Any) -> None:
        for key, value in changes.items():
            if isinstance(value, dict) and isinstance(self.data.get(key), dict):
                self.data[key].update(value)
            else:
                self.data[key] = value
        self.data["updated_at"] = _now()
        self.write()

    def feature(self, name: str, status: str) -> None:
        statuses = dict(self.data["feature_status"])
        statuses[name] = status
        self.update(feature_status=statuses)

    def write(self) -> None:
        temp = self.path.with_suffix(".json.tmp")
        content = json.dumps(self.data, ensure_ascii=False, indent=2)
        for attempt in range(5):
            try:
                temp.write_text(content, encoding="utf-8")
                temp.replace(self.path)
                return
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        try:
            self.path.write_text(content, encoding="utf-8")
        except PermissionError:
            # A transient reader lock must not terminate feature extraction or training.
            pass

    def complete(self, actual_rows: dict[str, int]) -> None:
        self.update(status="completed", current_stage="completed", actual_output_rows=actual_rows)
        self.write_markdown()

    def fail(self, stage: str, exc: BaseException) -> None:
        self.update(
            status="failed",
            current_stage=stage,
            error_stage=stage,
            error_message=f"{type(exc).__name__}: {exc}",
        )
        self.write_markdown()

    def write_markdown(self) -> None:
        middle = self.data["middle_embedding"]
        late = self.data["late_features"]
        model = self.data["model_progress"]
        actual = self.data.get("actual_output_rows", {})
        lines = [
            "# stagev3 run progress",
            "",
            f"1. Run mode: `{self.data['run_mode']}`",
            f"2. CV mode: `{self.data['cv_mode']}`",
            f"3. Feature extraction summary: `{json.dumps(self.data['feature_status'])}`",
            (
                "4. Cache reuse summary: "
                f"middle={middle['cached_windows']}/{middle['total_windows']}, "
                f"late={late['cached_transcripts']}/{late['total_transcripts']}"
            ),
            (
                "5. Number of API calls: "
                f"embedding_batches={middle['completed_batches']}, "
                f"late_requests={late['completed_requests']}"
            ),
            (
                "6. Number of completed model runs: "
                f"{model['completed_model_runs']}/{model['expected_model_runs']}"
            ),
            (
                "7. Expected vs actual output rows: "
                f"expected_model_runs={model['expected_model_runs']}, actual={json.dumps(actual)}"
            ),
            f"8. Failed stage: `{self.data.get('error_stage', 'none')}`",
        ]
        if self.data["run_mode"] == "selected_after_seed2026":
            lines.extend([
                f"9. Selected model count: `{self.data.get('selected_model_count', 0)}`",
                f"10. Total selected runs: `{self.data.get('total_selected_runs', 0)}`",
                f"11. Current seed: `{self.data.get('current_seed')}`",
                (
                    "12. Current selected model index: "
                    f"`{self.data.get('current_selected_model_index', 0)}`"
                ),
                f"13. Automatic workers: `{self.data.get('worker_count', 0)}`",
                f"14. Resume enabled: `{self.data.get('resume')}`",
                f"15. Force rerun: `{self.data.get('force_rerun')}`",
                f"16. Checkpoint directory: `{self.data.get('checkpoint_dir')}`",
                (
                    "17. Completed seeds: "
                    f"`{self.data.get('completed_seed_count', 0)}/"
                    f"{self.data.get('expected_seed_count', 0)}`"
                ),
                f"18. Missing seed IDs: `{self.data.get('missing_seed_ids', [])}`",
            ])
        if "error_message" in self.data:
            lines.append(f"   Error: `{self.data['error_message']}`")
        self.md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
