from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Style settings
import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


def interpolate(points):
    if len(points) <= 3:
        raise ValueError("At least 3 points are needed to interpolate")

    x_coords_2, y_coords_2 = zip(*points[:3])

    # np function to find the coefficients of the polynomial
    coefficients_deg_2 = np.polyfit(x_coords_2, y_coords_2, 2)
    polynomial_deg_2 = np.poly1d(coefficients_deg_2)

    x_coords_3, y_coords_3 = zip(*points[:4])
    coefficients_deg_3 = np.polyfit(x_coords_3, y_coords_3, 3)
    polynomial_deg_3 = np.poly1d(coefficients_deg_3)

    x = np.linspace(0, points[-1][0], 400)
    y_deg_2 = polynomial_deg_2(x)
    y_deg_3 = polynomial_deg_3(x)

    plt.figure(figsize=(8, 6))

    plt.plot(x, y_deg_2, 'g-', label='y = x^2')
    plt.plot(x, y_deg_3, 'b-', label='y = x^3')
    # plt.plot(x, y__e, 'r-', label='y = e^x')

    for point in points[3:]:
        plt.plot(point[0], point[1], 'kx', markersize=5)

    plt.legend()

    plt.show()


def compare_exact_and_heuristic_results(folder_workplace, experiment_name, csv_path_exact, csv_path_heuristic):

    e_marker = "o--"
    e_color = "g"

    h_marker = ">:"
    h_color = "r"

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
        fig1b_exact_data.append(np.mean(exact_results_n["makespan"].values))

        for R in exact_R_values:
            exact_results_n_R = exact_results_n[exact_results_n["R"] == R]
            fig2a_exact_data[exact_R_values.index(R)].append(np.mean(exact_results_n_R["duration"].values))
            fig2b_exact_data[exact_R_values.index(R)].append(np.mean(exact_results_n_R["makespan"].values))

    fig1a_heuristic_data = []
    fig1b_heuristic_data = []
    fig2a_heuristic_data = [[] for _ in range(len(heuristic_R_values))]
    fig2b_heuristic_data = [[] for _ in range(len(heuristic_R_values))]
    for n in heuristic_n_values:

        heuristic_results_n = heuristic_results[heuristic_results["n"] == n]
        fig1a_heuristic_data.append(np.mean(heuristic_results_n["duration"].values))
        fig1b_heuristic_data.append(np.mean(heuristic_results_n["makespan"].values))

        for R in heuristic_R_values:
            heuristic_results_n_R = heuristic_results_n[heuristic_results_n["R"] == R]
            fig2a_heuristic_data[heuristic_R_values.index(R)].append(np.mean(heuristic_results_n_R["duration"].values))
            fig2b_heuristic_data[heuristic_R_values.index(R)].append(np.mean(heuristic_results_n_R["makespan"].values))

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    ax[0, 0].set_title("Exact vs Heuristic DURATION")
    ax[0, 0].plot(exact_n_values, fig1a_exact_data, e_marker, color=e_color, label="Exact")
    ax[0, 0].plot(heuristic_n_values, fig1a_heuristic_data, h_marker, color=h_color, label="Heuristic")
    ax[0, 0].set_xticks(heuristic_n_values)
    ax[0, 0].legend()

    ax[0, 1].set_title("Exact vs Heuristic MAKESPAN")
    ax[0, 1].plot(exact_n_values, fig1b_exact_data, e_marker, color=e_color, label="Exact")
    ax[0, 1].plot(heuristic_n_values, fig1b_heuristic_data, h_marker, color=h_color, label="Heuristic")
    ax[0, 1].set_xticks(heuristic_n_values)
    ax[0, 1].legend()

    for i, R in enumerate(exact_R_values):

        ax[1, 0].plot(exact_n_values, fig2a_exact_data[i], e_marker, color=colors[i], label=f"Exact: R = {R}")
        ax[1, 0].plot(heuristic_n_values, fig2a_heuristic_data[i], h_marker, color=colors[i],
                      label=f"Heuristic: R = {R}")

        ax[1, 1].plot(exact_n_values, fig2b_exact_data[i], e_marker, color=colors[i], label=f"Exact: R = {R}")
        ax[1, 1].plot(heuristic_n_values, fig2b_heuristic_data[i], h_marker, color=colors[i],
                      label=f"Heuristic: R = {R}")

    ax[1, 0].set_title("Exact vs Heuristic DURATION, separated by R")
    ax[1, 0].set_xticks(heuristic_n_values)
    ax[1, 0].legend()

    ax[1, 1].set_title("Exact vs Heuristic MAKESPAN, separated by R")
    ax[1, 1].set_xticks(heuristic_n_values)
    ax[1, 1].legend()

    plt.savefig(folder_workplace + "/" + experiment_name, bbox_inches='tight', transparent=False, dpi=600)
    plt.show()


if __name__ == "__main__":
    # p0 = (0, 0)
    # p1 = (10, 0.14)
    # p2 = (20, 2.45)
    # p3 = (30, 15.65)
    # p4 = (50, 218.35)
    #
    # points = [p0, p1, p2, p3, p4]
    #
    # interpolate(points)

    folder = "plots_test_1/"
    compare_exact_and_heuristic_results(folder, "img_test_1", "exact_1.csv", "heuristic_1.csv")
