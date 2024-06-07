from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Style settings
import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


def exp_func(x, a, b):
    return a * np.exp(b * x)


def interpolate(folder_workplace, exp_name, csv_path_exact, csv_path_heuristic):
    exact_results = pd.read_csv(f"{folder_workplace}/{csv_path_exact}")
    heuristic_results = pd.read_csv(f"{folder_workplace}/{csv_path_heuristic}")

    # Calcola la media dei punti per ciascun valore di 'n'
    exact_avg = exact_results.groupby('n')['duration'].mean().reset_index()
    heuristic_avg = heuristic_results.groupby('n')['duration'].mean().reset_index()

    exact_points = list(zip(exact_avg["n"], exact_avg["duration"]))
    heuristic_points = list(zip(heuristic_avg["n"], heuristic_avg["duration"]))

    def interpolate_and_plot(points, title, img_name):
        if len(points) <= 3:
            raise ValueError("At least 3 points are needed to interpolate")

        x_coords_2, y_coords_2 = zip(*points[:3])

        coefficients_deg_2 = np.polyfit(x_coords_2, y_coords_2, 2)
        polynomial_deg_2 = np.poly1d(coefficients_deg_2)

        x_coords_3, y_coords_3 = zip(*points[:4])
        coefficients_deg_3 = np.polyfit(x_coords_3, y_coords_3, 3)
        polynomial_deg_3 = np.poly1d(coefficients_deg_3)

        # Usa solo i primi 3 punti per l'interpolazione esponenziale
        x_coords_exp, y_coords_exp = zip(*points[:3])
        x_data_exp = np.array(x_coords_exp)
        y_data_exp = np.array(y_coords_exp)
        popt_exp, _ = curve_fit(exp_func, x_data_exp, y_data_exp)

        x = np.linspace(min([p[0] for p in points]), max([p[0] for p in points]), 400)
        y_deg_2 = polynomial_deg_2(x)
        y_deg_3 = polynomial_deg_3(x)
        y_exp = exp_func(x, *popt_exp)

        plt.figure(figsize=(10, 5))
        plt.plot(x, y_deg_2, 'g-', label='y = x^2')
        plt.plot(x, y_deg_3, 'b-', label='y = x^3')
        plt.plot(x, y_exp, 'r-', label='y = e^x')

        for point in points[3:]:
            plt.plot(point[0], point[1], 'kx', markersize=5)

        plt.scatter(*zip(*points[:3]), label='Used points', color='blue')
        plt.scatter(*zip(*points[3:]), label='Unused points', color='purple')

        plt.title(title)
        plt.legend()
        plt.savefig(folder_workplace + "/" + img_name, bbox_inches='tight', transparent=False, dpi=600)
        plt.show()

    interpolate_and_plot(exact_points, 'Exact Results Interpolation', exp_name + "_exact")
    interpolate_and_plot(heuristic_points, 'Heuristic Results Interpolation', exp_name + "_heuristic")


def compare_exact_and_heuristic_results(folder_workplace, experiment_name, csv_path_exact, csv_path_heuristic):
    e_marker = "o--"
    e_color = "g"

    h_marker = ">:"
    h_color = "r"

    difference_marker = "-"
    difference_color = "k"
    colors = ['b', 'g', 'r']

    exact_results = pd.read_csv(folder_workplace + "/" + csv_path_exact, header=0, sep=",")
    heuristic_results = pd.read_csv(folder_workplace + "/" + csv_path_heuristic, header=0, sep=",")

    exact_n_values = sorted(exact_results["n"].unique())
    exact_R_values = sorted(exact_results["R"].unique())
    exact_runs = sorted(exact_results["run"].unique())

    heuristic_n_values = sorted(heuristic_results["n"].unique())
    heuristic_R_values = sorted(heuristic_results["R"].unique())
    heuristic_runs = sorted(heuristic_results["run"].unique())

    if exact_R_values != heuristic_R_values:
        raise ValueError("Different R values in exact and heuristic results")

    # fig1a: exact vs. heuristic duration, considering the Rs all together
    # fig1b: as fig1a, but showing makespan instead of duration
    # fig2a: exact vs. heuristic duration considering the Rs separately
    # fig2b: as fig2a, but showing makespan instead of duration

    fig1a_exact_data = []
    fig1b_exact_data = []
    fig2a_exact_data = [[] for _ in range(len(exact_R_values))]
    fig2b_exact_data = [[] for _ in range(len(exact_R_values))]
    for n in exact_n_values:

        exact_results_n = exact_results[exact_results["n"] == n]
        fig1a_exact_data.append(np.mean(exact_results_n["duration"].values))
        fig1b_exact_data.append(np.std(exact_results_n["duration"].values))

        for R in exact_R_values:
            exact_results_n_R = exact_results_n[exact_results_n["R"] == R]
            fig2a_exact_data[exact_R_values.index(R)].append(np.mean(exact_results_n_R["duration"].values))
            fig2b_exact_data[exact_R_values.index(R)].append(np.std(exact_results_n_R["duration"].values))

    fig1a_heuristic_data = []
    fig1b_heuristic_data = []
    fig2a_heuristic_data = [[] for _ in range(len(heuristic_R_values))]
    fig2b_heuristic_data = [[] for _ in range(len(heuristic_R_values))]
    for n in heuristic_n_values:

        heuristic_results_n = heuristic_results[heuristic_results["n"] == n]
        fig1a_heuristic_data.append(np.mean(heuristic_results_n["duration"].values))
        fig1b_heuristic_data.append(np.std(heuristic_results_n["duration"].values))

        for R in heuristic_R_values:
            heuristic_results_n_R = heuristic_results_n[heuristic_results_n["R"] == R]
            fig2a_heuristic_data[heuristic_R_values.index(R)].append(np.mean(heuristic_results_n_R["duration"].values))
            fig2b_heuristic_data[heuristic_R_values.index(R)].append(np.std(heuristic_results_n_R["duration"].values))

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))

    ax[0, 0].set_title("Exact vs. Heuristic", fontsize=12)
    ax[0, 0].plot(exact_n_values, fig1a_exact_data, e_marker, color=e_color, label="Exact")
    ax[0, 0].plot(heuristic_n_values, fig1a_heuristic_data, h_marker, color=h_color, label="Heuristic")
    ax[0, 0].set_xticks(heuristic_n_values)
    ax[0, 0].set_xlabel(r"$n_\text{jobs}$")
    ax[0, 0].set_ylabel(r"Computation Time [s]", fontsize=10, fontweight='bold')
    ax[0, 0].legend()

    ax[0, 1].set_title("Exact vs. Heuristic", fontsize=12)
    ax[0, 1].plot(exact_n_values, fig1b_exact_data, e_marker, color=e_color, label="Exact")
    ax[0, 1].plot(heuristic_n_values, fig1b_heuristic_data, h_marker, color=h_color, label="Heuristic")
    ax[0, 1].set_xticks(heuristic_n_values)
    ax[0, 1].set_xlabel(r"$n_\text{jobs}$")
    ax[0, 1].set_ylabel(r"Duration Std [s]", fontsize=10, fontweight='bold')
    ax[0, 1].legend()

    for i, R in enumerate(heuristic_R_values):

        ax[1, 0].plot(exact_n_values, fig2a_exact_data[i], e_marker, color=colors[i], label=f"Exact: R = {R}")
        #
        # if R in exact_R_values:
        ax[1, 0].plot(heuristic_n_values, fig2a_heuristic_data[i], h_marker, color=colors[i],
                      label=f"Heuristic: R = {R}")

        ax[1, 1].plot(exact_n_values, fig2b_exact_data[i], e_marker, color=colors[i], label="Exact")
        ax[1, 1].plot(heuristic_n_values, fig2b_heuristic_data[i], h_marker, color=colors[i], label="Heuristic")
        # ax[1, 1].plot(heuristic_n_values, fig2b_heuristic_data[i], h_marker, color=colors[i],
        #               label=f"Heuristic: R = {R}")

    ax[1, 0].set_title("Exact vs. Heuristic, separated by R", fontsize=12)
    ax[1, 0].set_xticks(heuristic_n_values)
    ax[1, 0].set_xlabel(r"$n_\text{jobs}$")
    ax[1, 0].set_ylabel(r"Computation Time [s]", fontsize=10, fontweight='bold')
    ax[1, 0].legend()

    ax[1, 1].set_title("Exact vs. Heuristic, separated by R", fontsize=12)
    ax[1, 1].set_xticks(heuristic_n_values)
    ax[1, 1].set_xlabel(r"$n_\text{jobs}$")
    ax[1, 1].set_ylabel(r"Duration Std [s]", fontsize=10, fontweight='bold')
    ax[1, 1].legend()

    plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
    plt.savefig(folder_workplace + "/" + experiment_name, bbox_inches='tight', transparent=False, dpi=600)
    plt.show()


if __name__ == "__main__":
    folder = "experiments_results"
    compare_exact_and_heuristic_results(folder,
                                        "img_test",
                                        "exact_results.csv",
                                        "heuristic_results.csv")

    # interpolate(folder,
    #             "img_test_2",
    #             "exact_results.csv",
    #             "heuristic_results.csv")
