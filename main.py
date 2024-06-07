from GenerateInstances import *
from PreProcess import PreProcess

if __name__ == "__main__":

    I_R = 3
    I_ILS = 100
    omega = 2
    N = 3
    gamma = 0.5

    n_jobs = 30

    working_folder = "experiments_results_1/"
    instance = GenerateInstances(n_jobs=n_jobs, dispersion=0.5, setup_times_interval=None, target_folder=working_folder)
    instance.export_csv()
    instance_name = instance.get_csv_name()
    instance_file_path = f"{working_folder}{instance_name}"

    pre_solver = PreProcess(instance_file_path)

    print(pre_solver.T)

    pre_solver = PreProcess(instance_file_path)

    print(pre_solver.T)