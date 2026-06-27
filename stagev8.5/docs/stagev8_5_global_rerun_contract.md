# Stagev8.5 global rerun contract

`--mode global_rerun` is the only one-command full experiment path.

- It clears only generated Stagev8.5 outputs under `output/` that can affect reproducibility.
- It retains raw input CSVs and all hash-locked Stagev5/Stagev4/Stagev6 reference assets.
- It starts M and L extraction with empty runtime caches and requires live API calls.
- It uses the Windows curl.exe/Schannel transport shim only to send the original request payloads; the copied feature source, preprocessing, windowing, cache keys, parsers, and feature aggregation remain unchanged.
- It verifies anchor parity, runs Stagev8.5 training, performs stability/bootstrapping, and executes the audit notebook.
- It writes `output/checks/stagev8_5_global_rerun_manifest.json` only after all phases complete.

Do not run `extract_features --force` or `train --force` separately when this global run is intended; those commands create partial reruns without the final global manifest.
