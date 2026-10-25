# Executed verification — 25 October 2026

## Preserved legacy case study

- Before upgrade: **63 original tests passed**, 33.03s.
- After upgrade: **63 original tests passed**, 18.10s. The warning is the intentional unseen-category case, not a failure.
- Original Streamlit application: **13 dashboard checks passed**, covering all eight pages and controls.
- The smoke check's regenerated verification JSON was copied to the new platform evidence folder; the original root artifact was restored exactly from its original Git blob. No legacy source, tests, documentation, original CSV or trained artifact was changed. The final commit changes only `platform/`.

Actual output: `benchmarks/legacy_tests_after.txt`, `legacy_dashboard_after.txt`, `legacy_dashboard_after.json`.

## Generic platform

- **104 backend tests passed**, 30.31s, no skips. Coverage includes ingestion/CSV/XLSX, positive encodings, duplicates and missingness, proxy/identifier guards, exact preprocessing isolation, all four models, reproducibility, actual inference/ensemble arithmetic, persisted versions, explanation fallback, ambiguous encoded-feature names, semantic evidence, protected attributes, budget formulas, API modes/errors/jobs/restart, exports and selected-model ranking.
- **18 frontend tests passed**, 9.01s in the captured final run. These use explicitly mocked test fixtures for upload/navigation, schema safety, API payloads, asynchronous context changes and job scoping. They are not presented as benchmark datasets.
- TypeScript checking and the Vite production build passed. The build reports a nonblocking approximately 792kB JavaScript bundle (234kB gzip); code splitting remains a future performance improvement.
- **36 additional live HTTP checks passed**, including reloading all persisted real benchmark models after server restart, actual SHAP, incompatible requests, genuine unlabelled exploratory mode, absent predictions in Mode C, and a real insufficient-data training job that fails with its actual diagnostic rather than fabricated metrics.

Actual output: `benchmarks/backend_tests.txt`, `frontend_tests.txt`, `frontend_build.txt`, `live_verification.json`.

## Real external datasets

All three requested sources were accessible without authentication during this run. Actual files were inspected and trained through the API: Iranian3150records, banking10000records, e-commerce5630records. A separately configured generic IBM7043record run also passed. Every dataset received its own saved schema, separate training job, four-model evaluation and held-out comparison. The full executed table and exclusions are in `benchmarks/results.md`; numeric results are in `results.json`.

The source mismatch in Iranian charge bands is disclosed; its pricing mapping remains unconfirmed. Status is excluded after a timing review. Bank and e-commerce horizon limitations remain explicit. Raw source files and detailed local per-row outputs are not committed. No synthetic replacement was used or reported as an external benchmark.

## Browser end-to-end and responsive checks

The built React app was tested against the actual local backend at **1440×900** and **390×844**. Both views fit their document width without horizontal page overflow. Navigation, all eight screens, real Dataset Studio upload/configuration/semantic confirmation, browser-started training/progress/completion, existing model diagnostics, SHAP, empty search, twelve-row unlabelled batch scoring/explanations, actual report download, constrained what-if and editable budget economics were exercised. The mobile drawer opened and closed after navigation. No JavaScript console errors were observed.

The browser training demonstration used an explicitly selected smaller grid and does not overwrite the standard-tuning benchmark results. The economic example (60 targets, 6 expected incremental retained, 440 net, 44% ROI) is an editable hypothetical scenario, not an observed financial result. Full detail is in `benchmarks/browser_verification.json`; actual screenshots are in `benchmarks/screenshots/`.

## Remaining scope limits

Independent customer snapshots and binary outcomes are supported. Conflicting repeated IDs, temporal/group training, multiclass targets, raw text/images/event streams, causal/uplift estimation and automated live retraining are not implemented. Constrained what-if is supported; DiCE counterfactual optimization is not included. Source timing, sample bias, small groups and unmeasured intervention effectiveness limit real-world claims. The app is local single-user software, not a multi-tenant production service.

These checks cover the declared supported cases. They do not establish correctness for every arbitrary file or prospective business population.
