from Heuristic import Heuristic
from Solver import Solver
from GenerateInstances import GenerateInstances
import pandas as pd

import csv

if __name__ == "__main__":
    import pandas as pd
    print(pd.__version__)

    I_R = 2
    I_ILS = 100
    omega = 2
    N = 3
    gamma = 0.5

    optimisation_timeLimit = 3600

    # Generate instances
    n_values = [4, 7, 10, 13, 16, 20, 30]
    R_values = [0.5, 1, 1.5]

    exact_n_values_range = range(5)

    runs = 5

    experiments = [[[] for __ in R_values] for _ in n_values]

    folder = "experiments_results"

    for i_n, n in enumerate(n_values):
        for i_R, R in enumerate(R_values):
            for _ in range(runs):
                print(f"Generating instance {n}n_{R}R...")
                instance = GenerateInstances(n_jobs=n, dispersion=R, target_folder=folder)
                instance.export_csv()
                experiments[i_n][i_R].append(f"{folder}/{instance.get_csv_name()}")

    print("\n\n")

    heuristic_results = []
    exact_results = []

    # Solve instances
    for i_n, n in enumerate(n_values):
        for i_R, R in enumerate(R_values):
            for run, instance_name in enumerate(experiments[i_n][i_R]):
                if i_n in exact_n_values_range:
                    print(f"Starting exact solver for instance {instance_name}...")
                    solver = Solver(instance_name)
                    solver.model.setParam('outputFlag', 0)
                    solver.model.setParam('timeLimit', optimisation_timeLimit)
                    solver.solve()
                    exact_solution_makespan = solver.get_solution_makespan()
                    exact_duration = round(solver.model.Runtime)
                    exact_duration_with_model_construction = round(solver.duration_with_model_construction)
                    print(f"Exact solution: {exact_solution_makespan} found in "
                          f"{exact_duration} seconds (plus {exact_duration_with_model_construction - exact_duration} "
                          f"for model construction)")
                    new_exact_row = {
                        "n": n, "R": R, "run": run, "makespan": exact_solution_makespan,
                        "duration": exact_duration,
                        "duration_with_model_construction": exact_duration_with_model_construction,
                        "solution": str(solver.get_solution_job_sequence())
                        .replace(", ", "->")
                        .replace("[", "")
                        .replace("]", "")
                    }
                    exact_results.append(new_exact_row)
                    with open(f"{folder}/exact_results.csv", mode='w', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=exact_results[0].keys())
                        writer.writeheader()
                        writer.writerows(exact_results)

                    print("\n\n")

                print(f"Starting heuristic for instance {instance_name}...")
                heuristic = Heuristic(instance_name)
                heuristic_solution_makespan, heuristic_solution = heuristic.ILS_BS(I_R, I_ILS, omega, N, gamma)
                heuristic_duration = round(heuristic.TIME_ils_bs)
                print(f"Heuristic solution: {heuristic_solution_makespan} found in {heuristic_duration} seconds")
                new_heuristic_row = {
                    "n": n, "R": R, "run": run, "makespan": heuristic_solution_makespan,
                    "duration": heuristic_duration,
                    "solution": str(heuristic_solution)
                    .replace(", ", "->")
                    .replace("[", "")
                    .replace("]", "")
                }
                heuristic_results.append(new_heuristic_row)
                with open(f"{folder}/heuristic_results.csv", mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=heuristic_results[0].keys())
                    writer.writeheader()
                    writer.writerows(heuristic_results)

                print("\n\n")
