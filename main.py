from GenerateInstances import *
from PreProcess import *
from Solver import *
from Heuristic import *

if __name__ == "__main__":

    I_R = 10
    I_ILS = 100
    omega = 2
    N = 3
    gamma = 0.5

    # print("Starting 10n_05R...")
    # heuristic_1 = Heuristic("10n_05R")
    # sol_1_makespan, sol_1 = heuristic_1.ILS_BS(I_R, I_ILS, omega, N, gamma)
    # print("10n_05R makespan: ", sol_1_makespan)
    #
    # print("\n\n")
    #
    # print("Starting 20n_05R...")
    # heuristic_2 = Heuristic("20n_05R")
    # sol_2_makespan, sol_2 = heuristic_2.ILS_BS(I_R, I_ILS, omega, N, gamma)
    # print("20n_05R makespan: ", sol_2_makespan)
    #
    # print("\n\n")
    #
    # print("Starting 30n_05R...")
    # heuristic_3 = Heuristic("30n_05R")
    # sol_3_makespan, sol_3 = heuristic_3.ILS_BS(I_R, I_ILS, omega, N, gamma)
    # print("30n_05R makespan: ", sol_3_makespan)

    print("Starting 50n_05R...")
    heuristic_5 = Heuristic("50n_05R")
    sol_5_makespan, sol_5 = heuristic_5.ILS_BS(I_R, I_ILS, omega, N, gamma)
    print("50n_05R makespan: ", sol_5_makespan)