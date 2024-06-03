from Heuristic import Heuristic
from Solver import Solver
from GenerateInstances import GenerateInstances
import pandas as pd

if __name__ == "__main__":

    print(pd.__version__)

    I_R = 10
    I_ILS = 100
    omega = 2
    N = 3
    gamma = 0.5

    # Generate instances
    n_values = [4, 8, 12, 16, 20, 24]
    R_values = [0.5, 1, 1.5]

    exact_n_values_range = range(4)

    runs = 3

    experiments = [[[] for __ in R_values] for _ in n_values]

    folder = "experiments_results"

    for i_n, n in enumerate(n_values):
        for i_R, R in enumerate(R_values):
            for _ in range(runs):
                print(f"Generating instance {n}n_{R}R...")
                instance = GenerateInstances(n, R)
                instance.export_csv()
                experiments[i_n][i_R].append(instance.get_csv_name())

    heuristic_results = pd.DataFrame(columns=["n", "R", "run", "makespan", "duration"])
    exact_results = pd.DataFrame(columns=["n", "R", "run", "makespan", "duration"])

    # Solve instances
    for i_n, n in enumerate(n_values):
        for i_R, R in enumerate(R_values):
            for run, instance_name in enumerate(experiments[i_n][i_R]):
                if i_n in exact_n_values_range:
                    print(f"Starting exact solver for instance {instance_name}...")
                    solver = Solver(instance_name)
                    solver.solve()
                    exact_solution_makespan = solver.get_solution_makespan()
                    exact_duration = solver.duration
                    print(f"Exact solution: {exact_solution_makespan} found in {exact_duration} seconds")
                    new_exact_row = pd.Series(data={"n": n, "R": R, "run": run, "makespan": exact_solution_makespan,
                                                    "duration": exact_duration})
                    exact_results = pd.concat([exact_results, pd.DataFrame([new_exact_row])], ignore_index=True)
                    exact_results.to_csv(f"{folder}/exact_results.csv", index=False)

                print(f"Starting heuristic for instance {instance_name}...")
                heuristic = Heuristic(instance_name)
                heuristic_solution_makespan, heuristic_solution = heuristic.ILS_BS(I_R, I_ILS, omega, N, gamma)
                heuristic_duration = heuristic.TIME_ils_bs
                print(f"Heuristic solution: {heuristic_solution_makespan} found in {heuristic_duration} seconds")
                new_heuristic_row = pd.Series(data={"n": n, "R": R, "run": run, "makespan": heuristic_solution_makespan,
                                                    "duration": heuristic_duration})
                heuristic_results = pd.concat([heuristic_results, pd.DataFrame([new_heuristic_row])], ignore_index=True)
                # heuristic_results = heuristic_results.concat(new_heuristic_row, ignore_index=True)
                heuristic_results.to_csv(f"{folder}/heuristic_results.csv", index=False)
