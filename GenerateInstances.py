import random
import numpy as np
import os


def randint_from_interval(interval):
    return random.randint(interval[0], interval[1])


class GenerateInstances:
    """
    Generation procedure described by Ovacikt et al. in:
        "Rolling horizon algorithms for a single-machine dybamic scheduling problem with sequence-dependent
         setup times"
        (1994) International Journal of Production Research
    """

    def __init__(self, n_jobs, dispersion, processing_times_interval=None,
                 setup_times_interval=None, target_folder="GeneratedInstances"):

        self.csv_name = None

        if processing_times_interval is None:
            processing_times_interval = [1, 100]
        if setup_times_interval is None:
            setup_times_interval = [1, 100]

        self.target_folder = target_folder
        self.n_jobs = n_jobs

        # The *_times_interval variables specify the range of values that the processing and setup times can assume
        self.processing_times_interval = processing_times_interval
        self.setup_times_interval = setup_times_interval

        # The dispersion is a parameter that influences the release dates distribution
        self.dispersion = dispersion
        avg_processing_time = (processing_times_interval[1] - processing_times_interval[0]) / 2
        self.release_dates_interval = [1, int(n_jobs * dispersion * avg_processing_time)]

        # Here we initialize all the data structures with -1, which is the value for not-allowable data
        # that we decided to include to have perfect match between each index and the respective job
        self.release_dates = np.full(self.n_jobs + 1, -1)
        self.processing_times = np.full(self.n_jobs + 1, -1)
        self.setup_times = np.full([self.n_jobs + 1, self.n_jobs + 1], -1, dtype=int)

        self.generate()

    # Generate the instances according to the aforementioned paper
    def generate(self):
        # For each job of the specified number of jobs
        for i in range(0, self.n_jobs + 1):

            # Generate randomly release date and processing time for each job (except dummy job)
            if i != 0:
                self.processing_times[i] = randint_from_interval(self.processing_times_interval)
                self.release_dates[i] = randint_from_interval(self.release_dates_interval)

            # Generate randomly setup time for each couple of jobs (except couple i -> i or i->0)
            for j in range(0, self.n_jobs + 1):
                # setup time i -> j
                if j != 0 and i != j:
                    self.setup_times[i, j] = randint_from_interval(self.setup_times_interval)

    # Specify the path where the folder have to be stored
    # The folder will contain 3 files csv, one for each datastructure
    def export_csv(self):
        parent_folder = os.getcwd().replace("\\", "/") + "/" + self.target_folder
        folder_name = "{:02d}".format(int(self.n_jobs)) + "n_" + "{:02d}".format(int(10 * self.dispersion)) + "R"
        if folder_name in os.listdir(parent_folder):
            i = 1
            while folder_name + str(f"_{i}") in os.listdir(parent_folder):
                i = i + 1
            folder_name = folder_name + str(f"_{i}")
        path = parent_folder + "/" + folder_name
        os.mkdir(path)
        np.savetxt(path + "/release_dates.csv", self.release_dates, delimiter=",", fmt='%i')
        np.savetxt(path + "/processing_times.csv", self.processing_times, delimiter=",", fmt='%i')
        np.savetxt(path + "/setup_times.csv", self.setup_times, delimiter=",", fmt='%i')
        print("Files exported in: ", path)
        self.csv_name = folder_name

    # Return the name of the folder where the csv files are stored (not the path)
    def get_csv_name(self):
        if self.csv_name is None:
            print("This instance has not been exported yet")
        else:
            return self.csv_name
