# stagev5 feature policy

- **E** strictly follows `stagev2.zip` early feature extraction, including its original preprocessing, training-only BM25 fit, information-unit definitions, and `early_v5_mild_sensitive` logic.
- **M** strictly follows `stagev2.zip` middle feature extraction, including its original tokenization, 15-word windows, stride 5, cache key, API call boundary, and embedding-dimension output structure.
- **L** strictly follows `stagev4_unmasked_form_comparator.zip` late P4 unmasked F8 extraction, including original unmasked transcript handling, strict F8 JSON schema, allowed 1/3/5/7/9 values, and cache semantics.

The adapter only aligns IDs, columns, manifests, and model blocks. It redirects output folders, validates IDs, and converts the P4 train/external file layout into the AD/control/test file layout expected by the stagev2 classifier core. It does not redefine E, M, or L.

`--mode train` reads existing E/M/L CSV files and does not call APIs. `--mode render_notebook` reads existing result files and does not call APIs. `--mode all` and `--mode extract_features` can trigger feature extraction and API usage.
