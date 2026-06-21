from __future__ import annotations


def unmasked_transcript(text: str) -> str:
    """Return the frozen normalized ASR transcript unchanged for P4 scoring."""
    return str(text).strip()
