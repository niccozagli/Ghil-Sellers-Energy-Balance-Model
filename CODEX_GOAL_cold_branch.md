# Codex goal: PlaSim–LSG cold branch — mechanism and Koopman path to snowball (autonomous run)

Read `AGENTS.md` and `Plasim_Snowball_Mechanism_Koopman_Plan.md` (the "Plan") before anything else.
Work autonomously through the whole Plan. The aim is to **answer the scientific questions below with
evidence**, not only to write code. Do not ask for confirmation; make reasonable choices, record them,
and continue.

Experiments root: `/Volumes/Nicco/Plasim/experiments`. μ in scope:
1265 (warm reference), 1245, 1240, 1235, 1230, 1225, 1216. Also make every script accept any other
`CONTROL_360ppm_T21L10_10000Y_MU_<μ>` without code changes.

## Questions to answer (the deliverable)

Q1. Is the ~40–80 yr SH oscillation physical? (energy closure by phase, 9 m cap artefact, window robustness)
Q2. What is the mechanism? Order and lags of the chain subtropical 200–750 m water → overturning/convection
    at the SH margin → ice edge → absorbed SW → surface T; is it a delayed oscillator with P ≈ 2τ; how do τ,
    its spread σ, and ocean heat convergence at the edge change with μ? Heat- or freshwater-driven convection?
Q3. How does the transition happen? Which hemisphere runs away first at 1225, how are NH and SH coupled
    (lags, sign, atmospheric compensation), and does NH land snow matter?
Q4. Do reduced RP resonances show the path to the bifurcation? Oscillatory pair (period, coherence) and
    leading real mode vs μ; which observables are needed for lag-consistent (Markovian) estimates; agreement
    with ACF fits and amplitude relaxation.
Q5. Can the transition be fingerprinted from the NH alone (two-timescale ACF tail, delay-embedded KDMD,
    projection on full-analysis modes)?
Q6. (If time) Does the delayed EBM (Plan 6.1) reproduce P(μ) and the loss of oscillation?

For each question give: answer (yes / no / partial), key numbers with uncertainty, the figure(s) that show
it, and confidence with the main caveat.

## How to work

- Follow the Plan's order (Section 8). Each task: library code in `src/gsebm/` + unit tests (synthetic data
  with known answers), Typer CLI in `scripts/`, run on the data, write outputs to `data/Plasim/<EXP>/` and
  `figures/plasim_cold_branch/`, then run `PYTHONPATH=src uv run python -m unittest discover -s tests`.
  Never overwrite existing outputs (use new filenames or a `--overwrite` flag that defaults to false).
- Commit after each completed task on a branch `codex/cold-branch` with a message naming the Plan section.
- Keep a running log `Plasim_Cold_Branch_Log.md`: what was done, choices made (thresholds, filters,
  variable codes found), timings, problems. Update it after every task.
- Keep `Plasim_Cold_Branch_Results.md` current: after each section, fill in the answer to the relevant
  question so a partial run is still useful.
- Use marimo apps `analysis/plasim_cold_branch_mechanism.py` (Sections 3–4) and
  `analysis/plasim_cold_branch_koopman.py` (Section 5) for the final figures; heavy computation goes in
  scripts, apps only load results.
- Performance: use `--workers` for per-file extraction; cache intermediate `.nc` files; for KDMD subsample
  consistently or use Nyström if N² memory is too large, and log it.

## Gates — do not stop, adapt

- Missing field in Task 0: log it, skip only the tasks that need it, mark the affected question as
  "not answerable with available output" and continue.
- 3.3 (cap artefact) or 3.2 (energy closure) fails: do NOT stop. Record it prominently in Results under Q1,
  continue the Plan, and add to every later result the caveat that the cycle may be partly an artefact.
  Additionally quantify how much of the cycle variance co-varies with the cap flux (regression) and repeat
  the key Koopman result (5.4) with the cap-flux-correlated part regressed out.
- Direct vs implied OHT mismatch > 20% (2.5): use implied OHT for Section 4, log the mismatch.
- Non-stationary runs (1230, 1225): windowed analysis only (Plan 5.5); never mix windows into one estimate.
- If a Koopman estimate is not lag-consistent, report it as unresolved; do not pick the lag that gives the
  nicest number. Report the full weight/lag scans in an appendix table.

## Scientific hygiene

- Remove the post-transient mean; if detrending, re-centre afterwards. State transients used per μ.
- Uncertainty everywhere: block bootstrap (200-yr blocks) or split halves; report intervals, not points.
- Validate every new method on a synthetic case with known answer before applying it
  (Hilbert phase: noisy sinusoid; delay fit: gamma-kernel linear system; KDMD: stochastic Hopf and noisy
  focus with analytic resonances; onset detection: synthetic ramp).
- Do not tune thresholds to get an expected result; if a result contradicts the hypothesis in the Plan's
  Section 0, report it as such — that is a valid answer.
- Sign conventions: TOA N positive downward (see existing code); scipy CSD phase positive = first variable
  leads (verify with a synthetic test).

## Finish

End with `Plasim_Cold_Branch_Results.md` containing: a one-paragraph summary; Q1–Q6 answers as above;
a table per μ (mean Ts, ice edges, AMOC/overturning, period, τ, σ, edge OHT convergence, resonances);
a list of figures; open issues and suggested next runs (e.g. which extra μ values would most sharpen the
bifurcation picture). Also apply the `Plasim_Diagnostics.md` corrections listed in Plan Section 7.
