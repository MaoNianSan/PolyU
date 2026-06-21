# stagev5 feature policy

- **E** is generated only by the copied `stagev2` early feature code, including its original preprocessing, training-only BM25 fit, information-unit definitions, and `early_v5_mild_sensitive` logic.
- **M** is generated only by the copied `stagev2` BGE-M3 code, including its original tokenization, 15-word windows, stride 5, cache key, API call, and embedding-dimension output structure.
- **L** is generated only by the copied `stagev4_unmasked_form_comparator` P4 implementation, including original unmasked transcript handling, strict F8 JSON schema, allowed 1/3/5/7/9 values, and cache semantics.

The adapter only redirects output folders, validates IDs, and converts the P4 train/external file layout into the AD/control/test file layout expected by the stagev2 classifier core. It does not change any E/M/L feature calculation.
