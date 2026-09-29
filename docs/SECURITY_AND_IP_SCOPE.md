# Security and IP scope

This publication **candidate** intentionally contains only: the deterministic synthetic connected-grid kernel; small CPU/CUDA smoke and unexecuted calibration harness; explicit S1/S2/S3 config; sanitized raw technical outputs; example Leonardo scripts; reproduction/claim-boundary documents; tests and CI. The kernel and synthetic constants disclose a compute fixture, not the private active-search system.

Intentionally excluded from the candidate:

- `data/sanitized/promotion_accounting.json` and all R1/R1.1/R1.2 private method, acquisition, diversity, promotion and solver evidence;
- all `*.sqlite`, `*.sqlite-wal`, `*.sqlite-shm`, review ZIPs, model checkpoints and private archives;
- Task A/B, CALIBRATION/EVALUATION, CINELDI, customer/DSO data and private source rows;
- private emails, credentials, tokens, `.env` files, machine paths and user identifiers;
- unpublished commercial architecture and scientific result interpretations.

Only an explicit file allowlist is staged. `.gitignore` is an additional guard, not the release boundary. `tools/check_public.py verify` checks the manifest, filenames and content; final `git ls-files` and diff inspection remain mandatory. This audit makes no legal patentability determination. The restricted [LICENSE](../LICENSE) preserves Negentra's published Q-CHARGE research/evaluation model, changing only its project and application context sentence to identify BLACKSWAN-GRID and the EuroHPC AI Factory Playground application. No private research evidence was added.
