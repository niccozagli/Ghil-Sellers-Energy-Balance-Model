# PlaSim oscillation irregularity: exploratory scripts (2026-10-07)

These are the scratch drivers behind `analysis/Koopman_irregularity.py`
(section 10 summarizes the methods and findings). They are exploratory and
untested; the maintained functions live in `src/gsebm/plasim_koopman_irregularity.py`.

Run from the repository root with `PYTHONPATH=src uv run python <script> <output dir>`.
Drivers that need archives not in `data/Plasim` copy them from
`/Volumes/Nicco/Plasim/extracted` into `<output dir>/stage`, one μ at a time,
and delete the copy afterwards. Outputs are pickles in `<output dir>`.

| Script | What it does |
|---|---|
| `run_all_mu.py` | Irregularity analysis for the seven μ; picks clean windows and reports corrupt year ranges |
| `coupling_all_mu.py` | Ice–ocean latitude–lag coupling, clock-error removal, regression maps |
| `coupling_robustness_and_budget.py` | Coupling robustness variants and the gyre heat budget |
| `state_comparison.py` | Koopman states Ts+θ, ice+θ, ice+Ts+θ compared |
| `coupled_residual_koopman.py` | Joint ice–ocean residual Koopman mode |
| `physics_read_maps.py` | Reads SIC and Ts maps per μ (needs `coup/` and `rob/` outputs) |
| `physics_cycle_diagnostics.py`, `physics_cycle_budget.py` | Ice cover per T21 row, cycle heat budget, cycle vs irregular ice–gyre relation. Note: phased on the Koopman ψ₁ clock |
| `archive_integrity_scan.py`, `archive_integrity_local_copy.sh` | Read-only archive integrity scan (via local copies) |
| `early_*.py` | First 1240 vs 1232.5 probes; superseded by the drivers above |

`physics_*` and `early_*` scripts expect the pickles of the other drivers at
the paths given on the command line; they were run in a temporary job folder.
