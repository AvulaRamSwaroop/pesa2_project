# Benchmark expansion and baseline verification notes

## Algorithm reference

Primary reference: D. W. Corne, N. R. Jerram, J. D. Knowles and M. J. Oates, “PESA-II: Region-based Selection in Evolutionary Multiobjective Optimization,” GECCO-2001, pp. 283–290. Publication record: https://research.manchester.ac.uk/en/publications/pesa-ii-region-based-selection-in-evolutionary-multiobjective-opt/

The algorithm code is retained as a **PESA-II-style baseline**. It has not been certified as a line-by-line reproduction of every paper detail. No research improvement is included. Keep this baseline separate from future experiments that modify selection, archive management, or variation operators.

## Implemented benchmarks

- ZDT1, ZDT2, ZDT3, ZDT4, ZDT6.
- DTLZ1–DTLZ7.
- Standard Fonseca-Fleming, Kursawe, Schaffer, and Viennet definitions.

ZDT4 uses the common original convention `x1 in [0,1]`, `x2..xn in [-5,5]`. Other libraries may provide a `[-10,10]` variant; results from different bounds must not be mixed.

ZDT6's reference front is sampled directly in objective space from `f2=1-f1^2`, beginning at the minimum attainable first objective (approximately 0.2807753191). Fonseca, Kursawe, and Viennet reference sets are deterministic decision-space sampling approximations, not exact analytical fronts. Schaffer uses an analytical reference front. DTLZ reference sets are sampled from known front geometry/manifolds; DTLZ7 is an approximation. Reference-set quality must be validated before treating IGD as authoritative.

## Explicitly unresolved items from the provided benchmark PDF

The following are intentionally **not implemented under guessed formulas**: Binh (1), Binh (3), Deb, Deb (2), Fonseca (2), Laumanns, Murata, Quagliarella, Rendon, Rendon (2), Schaffer (2), Srinivas, Tanaka, and Viennet (2). Their exact equations, variable bounds, constraint definitions, and/or numbered variant identities must be confirmed from the original source. See `benchmark_catalog.csv` for each status.

Standard Srinivas and Tanaka formulations are constrained problems. The current optimizer does not implement feasibility-first constraint handling, so they should not be run as ordinary unconstrained objectives. The benchmark names “Binh (1)” and “Binh (3)” are not silently equated with the standard Binh-Nguyen (BNH) problem.

## Metrics and experimental limits

- HV is implemented only for two-objective fronts. Fixed reference points must be checked against each benchmark's objective range before final reporting.
- IGD/GD use unnormalized Euclidean distance; compare only with identical objective scaling and reference sets.
- A plot or passing unit test does not prove convergence or paper fidelity.
- Use identical problem definitions, seeds, evaluation budgets, and reference sets when comparing PESA-II, NSGA-II, and SPEA2.

## Before research claims

1. Complete paper-to-code review of grid construction, box selection probabilities, archive update/truncation, internal population semantics, variation operators, duplicate handling and evaluation budget.
2. Validate every reference set and HV reference point.
3. Add exact-source implementations and tests for each unresolved legacy variant before including it in experiments.
4. Run independent seeds and report mean and standard deviation.
