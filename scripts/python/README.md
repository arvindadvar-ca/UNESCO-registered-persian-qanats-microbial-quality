This folder contains Python scripts used for probabilistic microbial risk assessment.

Python was used to:
- Perform Monte Carlo simulations (10,000 iterations) to account for variability and uncertainty
- Generate probabilistic risk distributions based on deterministic exposure inputs
- Implement a secondary Monte Carlo simulation in which the β-Poisson dose–response parameters (α and N₅₀) are modeled by treating them as stochastic variables rather than fixed constants
- Export simulation outputs for further analysis

All Python scripts read input data from the `data/processed` directory and write results to the `results` directory.
