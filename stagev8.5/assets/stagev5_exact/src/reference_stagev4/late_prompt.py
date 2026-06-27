from __future__ import annotations

import hashlib
import json

RAW8_COLUMNS = [
    "late_sentence_structural_integrity",
    "late_phrase_continuity",
    "late_repetition_control",
    "late_repair_efficiency",
    "late_filler_control",
    "late_referential_clarity",
    "late_grammatical_stability",
    "late_local_coherence",
]
JSON_KEYS = [c.replace("late_", "") for c in RAW8_COLUMNS]

SYSTEM = """You are a criterion-referenced rater of observable expressive-form integrity in an unmasked English ASR transcript. You are not a diagnostician. Do not infer Alzheimer disease, cognitive status, or picture-content correctness. Return only the requested JSON object."""

RUBRIC = """
The input is an UNMASKED English ASR transcript. It may contain Cookie-Theft picture-description content, naming errors, incomplete picture coverage, generic wording, or unusual scene descriptions.

FORM-ONLY REFERENCE STANDARD
- Score only observable expression form. Do NOT score picture-content accuracy, number of picture events mentioned, lexical richness, amount of information, or whether expected characters, objects, or actions are named.
- An unmasked transcript can receive 7 or 9 even if it omits expected scene details or gives an inaccurate scene description, provided its expression form is stable.
- Do NOT lower a score solely for one isolated misspelling, fused token, truncated token, phonetic ASR artifact, rare word, or unusual content word.
- Count a problem only when it either affects a clause-like unit, recurs in at least two local units, or makes local expression difficult to recover.
- Score the eight dimensions independently. Do not copy one global impression across dimensions. There is no overall-quality score.

USE ONLY THESE VALUES
9 = stable and intact: no repeated criterion-relevant problem.
7 = mostly intact: one or two isolated, recoverable problems.
5 = mixed: several local problems, but most expression remains recoverable.
3 = clearly impaired: recurrent problems that affect local interpretation.
1 = persistently impaired: the criterion is not maintained across much of the transcript.

DIMENSION-SPECIFIC EVIDENCE
1. sentence_structural_integrity: score down only for repeated clause-like units lacking a recoverable predicate or core structure; content omissions or inaccurate scene statements do not count.
2. phrase_continuity: score down only for recurrent word sequences that cannot be connected into an interpretable local phrase; unusual scene vocabulary alone does not count.
3. repetition_control: score down only for uncontrolled immediate repetition of the same word or short phrase in a local window; meaningful re-mention of a person, object, or action does not count.
4. repair_efficiency: score down only when repeated false starts, abandoned starts, or repairs interrupt the expression; one recoverable restart is minor.
5. filler_control: score down only for visible accumulation of hesitation fillers such as uh, um, well, you know, or I mean; absence of fillers merits 9.
6. referential_clarity: score down only when a pronoun or noun reference cannot be tracked from the current or previous two local units; a generic or incorrect scene reference alone does not count.
7. grammatical_stability: score down only for recurrent function-word, agreement, order, or tense disruptions that alter a local proposition; a single ASR word error is not grammar evidence.
8. local_coherence: score down only when adjacent clause-like units repeatedly lack a recoverable local relation; missing or incorrect picture content is not a coherence problem.
""".strip()


def build_messages(transcript: str) -> list[dict[str, str]]:
    schema = {key: "one of 1, 3, 5, 7, 9" for key in JSON_KEYS}
    user = (
        f"{RUBRIC}\n\n"
        "First assess each criterion privately and independently. Then return only valid JSON with exactly these keys and no prose:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"Transcript:\n{transcript}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def prompt_hash() -> str:
    material = SYSTEM + "\n" + RUBRIC + "\n" + "|".join(JSON_KEYS)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
