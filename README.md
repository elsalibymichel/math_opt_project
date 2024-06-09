# Paper Reproduction
### [Exact and heuristic algorithms for minimizing the makespan on a single machine scheduling problem with sequence-dependent setup times and release dates](https://www.sciencedirect.com/science/article/abs/pii/S0377221723008652)
#### Morais et al. (2024) European Journal of Operational Research

The paper was reproduced for the exam of Mathematical Optimisation (2024, Prof. Lorenzo Castelli), at the University of Trieste

The code is extensively commented, and references for algorithms not presented in the main paper are cited in the code.<br />
Here we describe the content of the git repository files.

- `GeneratedInstances.py` generates data for testing our implementations
  - The provided instances from the paper required too much execution time for our purposes
- `PreProcess.py` pre-processes data for the exact and heuristic procedures
  - In particular, for the exact procedure
    - It sets the mathematical formulation (arcs and nodes) 
    - It sets an upper bound on the makespan using the Beam Search algorithm in `BeamSearch.py`
- `Solver.py` implements the exact procedure using Gurobi (we own the academic license)
- `Heuristic.py` implements the heuristic procedure
  - It uses the Beam Search algorithm from `BeamSearch.py`
- `test_algs.py` executes the exact and heuristic algorithms
  - It generates the instances using `GeneratedInstances.py`
  - It saves the results in two `.csv` files
    - `exact_results.csv`
    - `heuristic_results.csv` 
  - Instances and results are stored in the folder `experiments_results`
