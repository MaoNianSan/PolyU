# Stagev5 Feature Policy

Stagev5 is an adapter and reporting layer around locked feature definitions. It does not redefine the E, M, or L feature families.

## Locked Sources

| Feature family | Locked source | Required interpretation |
|---|---|---|
| E | `stagev2.zip` | E strictly follows stagev2 early feature extraction. |
| M | `stagev2.zip` | M strictly follows stagev2 middle feature extraction. |
| L | `stagev4_unmasked_form_comparator.zip` | L strictly follows stagev4 unmasked late P4/F8 extraction. |

The adapter only aligns IDs, columns, manifests, and model blocks; it does not redefine E, M, or L.

## Expected Feature Counts

- E model features: 61
- M model features: 1024
- L raw F8 model features: 8
- L auxiliary diagnostic features: displayed when present, but excluded from model inputs

## Command Boundary

`--mode train` only reads existing E/M/L CSV files and does not call APIs. `--mode render_notebook` only reads existing CSV, JSON, Markdown, and PNG result files and does not call APIs. `--mode all` and `--mode extract_features` can trigger feature extraction and API calls.

## Reporting Boundary

Do not edit CSV, JSON, Markdown, or PNG result artifacts to change metrics, predictions, confusion matrices, CIs, model ranking, subgroup performance, or error analysis. Any result change should come only from an explicitly requested rerun.
