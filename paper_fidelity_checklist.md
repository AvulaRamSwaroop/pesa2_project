# Paper fidelity checklist

Before claiming exact replication of Corne et al. (2001), verify from the full paper:
- archive merge and update semantics
- duplicate handling and dominance rules
- grid/hyperbox construction and boundary behavior
- sparse-region selection probabilities
- individual selection within a selected region
- archive truncation when over capacity
- variation operators and their parameters
- stopping budget and function-evaluation accounting
- benchmark-specific variables, bounds, and reference fronts
- common normalization, IGD reference set, and HV reference point for comparisons
- independent runs, random seeds, hardware/software environment

This repository is a reproducible baseline scaffold, not a certified exact replication. Preserve it unchanged when adding a research variant.
