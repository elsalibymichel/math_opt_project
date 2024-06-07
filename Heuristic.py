from PreProcess import PreProcess
from BeamSearch import beam_search

import random
import time


class Heuristic:
    """
    Heuristic class that implements the ILS-BS algorithm presented by Morais et al. in:
        "Exact and heuristic algorithms for minimizing the makespan on a single
        machine scheduling problem with sequence-dependent setup times and
        release dates"
        (2024) European Journal of Operational Research
    """
    def __init__(self, file_path):
        self.data = PreProcess(file_path)
        self.all_jobs = list(range(1, self.data.n_jobs + 1))

        # Some variables to track time spent in each function
        self.TIME_bs = 0
        self.TIME_swap = 0
        self.TIME_l_block = 0
        self.TIME_perturbation = 0
        self.TIME_ils_bs = 0

    # Class that define a subsequences of jobs
    # The methods of this class allow to reduce the time complexity of the ILS-BS algorithm
    class Subsequence:
        def __init__(self, jobs, processing_times, setup_times, release_dates):
            self.F = None
            self.L = None
            self.E = None
            self.D = None
            self.jobs = jobs

            self.processing_times = processing_times
            self.setup_times = setup_times
            self.release_dates = release_dates

            # Initialize the sequence
            # Due to the recursive definition of some variables (self.E, self.D, and self.C), the initialization
            # acts differently based on the number of jobs in the sequence
            if len(jobs) == 1:
                if jobs[0] == 0:
                    self.init_dummy_subsequence()
                    return
                else:
                    self.private_init_one_job(jobs[0])
                    return
            else:
                # If subsequence contains more than one job, the initialization is done by concatenating recursively
                # concatenating subsequences
                self.private_init_one_job(jobs[0])

                cumulative_subsequence = None
                for i in range(1, len(jobs)):
                    next_subsequence = Heuristic.Subsequence(
                        [jobs[i]],
                        self.processing_times,
                        self.setup_times,
                        self.release_dates,
                    )
                    cumulative_subsequence = self.concatenate_to(next_subsequence)

                    self.F = cumulative_subsequence.F
                    self.L = cumulative_subsequence.L
                    self.E = cumulative_subsequence.E
                    self.D = cumulative_subsequence.D
                return

        # Return a subsequence that contains only the dummy job
        def dummy_subsequence(self):
            return Heuristic.Subsequence(
                [0], self.processing_times, self.setup_times, self.release_dates
            )

        # Initialize a dummy subsequence
        def init_dummy_subsequence(self):
            self.F = 0
            self.L = 0
            self.E = 0
            self.D = 0
            return

        # Initialize a subsequence with only one job
        def private_init_one_job(self, job):
            self.F = job
            self.L = job
            self.E = self.release_dates[job]
            self.D = self.processing_times[job]
            return

        # Return a subsequence that is the concatenation between self subsequence and argument subsequence
        # (with this order) updating the variables F, L, E, D of the returned subsequence
        # according to the rules defined in the paper
        def concatenate_to(self, subseq2):
            new_jobs = self.jobs + subseq2.jobs
            new_F = self.F
            new_L = subseq2.L
            I = 0
            i, j = self.L, subseq2.F
            if self.C() < subseq2.E:
                if self.C() < self.release_dates[j]:
                    I = self.release_dates[j] - self.C()
                if self.C() + I + self.setup_times[i][j] < subseq2.E:
                    I = subseq2.E - (self.C() + self.setup_times[i][j])
            new_E = self.E + I
            new_D = self.D + subseq2.D + self.setup_times[i][j]

            new_subseq = self.copy()

            new_subseq.F = new_F
            new_subseq.L = new_L
            new_subseq.E = new_E
            new_subseq.D = new_D
            new_subseq.jobs = new_jobs

            return new_subseq

        # Return the completion time of the subsequence
        def C(self):
            return self.E + self.D

        # Return the makespan of the subsequence
        # This method can be used only if the subsequence contains all the jobs
        def makespan(self):
            if len(self.jobs) != len(self.release_dates) - 1:
                raise Exception("Seqence does not contain all jobs")
            else:
                dummy = self.dummy_subsequence()
                complete_solution = dummy.concatenate_to(self)
                makespan = complete_solution.C()
                return makespan

        # Return a copy of the subsequence
        def copy(self):
            new_instance = self.__class__.__new__(self.__class__)
            new_instance.jobs = self.jobs[:]
            new_instance.processing_times = self.processing_times
            new_instance.setup_times = self.setup_times
            new_instance.release_dates = self.release_dates
            new_instance.F = self.F
            new_instance.L = self.L
            new_instance.E = self.E
            new_instance.D = self.D
            return new_instance

    # Solution is a list of jobs ##########################################################################################
    def get_best_from_swap_neighborhood(self, solution: list):

        self.TIME_swap = self.TIME_swap - time.time()

        best_neighbor = None
        best_makespan = float("inf")
        for i in range(len(solution) - 2):
            job_prev_i = solution[:i]
            if i != 0:
                sub_seq_prev_i = Heuristic.Subsequence(
                    job_prev_i,
                    self.data.processing_times,
                    self.data.setup_times,
                    self.data.release_dates,
                )

            for j in range(i + 1, len(solution) - 1):
                jobs = solution.copy()
                jobs[i], jobs[j] = jobs[j], jobs[i]
                sub_seq_post_i = Heuristic.Subsequence(
                    jobs[i:],
                    self.data.processing_times,
                    self.data.setup_times,
                    self.data.release_dates,
                )

                makespan = None
                if i == 0:
                    makespan = sub_seq_post_i.makespan()
                else:
                    makespan = sub_seq_prev_i.concatenate_to(sub_seq_post_i).makespan()

                if makespan < best_makespan:
                    best_makespan = makespan
                    best_neighbor = jobs

        self.TIME_swap = self.TIME_swap + time.time()

        return best_makespan, best_neighbor

    def get_best_from_l_block_neighborhood(self, l, solution: list):

        self.TIME_l_block = self.TIME_l_block - time.time()

        best_neighbor = None
        best_neighbor_makespan = float("inf")
        for i in range(len(solution) - l + 1):
            sub_seq_i_l = Heuristic.Subsequence(
                solution[i: i + l],
                self.data.processing_times,
                self.data.setup_times,
                self.data.release_dates,
            )
            legit_indexes = [
                k for k in range(len(solution)) if k not in range(i, i + l)
            ]
            for j in legit_indexes:
                if j >= i + l:
                    prev_block = solution[:i] + solution[i + l: j + 1]
                    block = solution[i: i + l]
                    post_block = solution[j + 1:]
                else:
                    prev_block = solution[:j]
                    block = solution[i: i + l]
                    post_block = solution[j:i] + solution[i + l:]

                # prev_block + block + post_block
                if prev_block:
                    sub_seq_prev = Heuristic.Subsequence(
                        prev_block,
                        self.data.processing_times,
                        self.data.setup_times,
                        self.data.release_dates,
                    )
                    if post_block:
                        sub_seq_post = Heuristic.Subsequence(
                            post_block,
                            self.data.processing_times,
                            self.data.setup_times,
                            self.data.release_dates,
                        )
                        makespan = (
                            sub_seq_prev.concatenate_to(sub_seq_i_l)
                            .concatenate_to(sub_seq_post)
                            .makespan()
                        )
                        # print(1, prev_block, block, post_block)
                    else:
                        makespan = sub_seq_prev.concatenate_to(sub_seq_i_l).makespan()
                        # print(2, prev_block, block)
                else:
                    sub_seq_post = Heuristic.Subsequence(
                        post_block,
                        self.data.processing_times,
                        self.data.setup_times,
                        self.data.release_dates,
                    )
                    if post_block:
                        makespan = sub_seq_i_l.concatenate_to(
                            sub_seq_post
                        ).makespan()
                        # print(3, block, post_block)
                    else:
                        raise Exception("l-block should not be l=n_jobs")

                if makespan < best_neighbor_makespan:
                    best_neighbor_makespan = makespan
                    best_neighbor = prev_block + block + post_block

        self.TIME_l_block = self.TIME_l_block + time.time()
        return best_neighbor_makespan, best_neighbor

    def local_search(self, solution):

        # in neighborhood_list, k=0 is related to the swap neighborhood
        # while k=l (l!=0) il related to the l-block neighborhood
        neighborhood_list = [k for k in range(self.data.n_jobs)]
        current_best_solution = solution
        current_best_makespan = Heuristic.Subsequence(
            solution, self.data.processing_times, self.data.setup_times, self.data.release_dates
        ).makespan()

        while neighborhood_list:

            neighborhood = random.choice(neighborhood_list)
            new_solution_makespan = None
            if neighborhood == 0:
                new_solution_makespan, new_solution = self.get_best_from_swap_neighborhood(current_best_solution)
            else:
                l = neighborhood
                new_solution_makespan, new_solution = self.get_best_from_l_block_neighborhood(l, current_best_solution)

            if new_solution_makespan < current_best_makespan:
                current_best_solution = new_solution
                current_best_makespan = new_solution_makespan

                # Update NL
                neighborhood_list = [k for k in range(self.data.n_jobs)]
            else:
                neighborhood_list.remove(neighborhood)

        return current_best_makespan, current_best_solution

    def Perturbation(self, solution):

        self.TIME_perturbation = self.TIME_perturbation - time.time()

        indexes_list = random.sample([k for k in range(0, self.data.n_jobs)], 4)
        indexes_list.sort()

        prev_seq = solution[: indexes_list[0]]
        sub_seq_1 = solution[indexes_list[0]: indexes_list[1] + 1]
        mid_seq = solution[indexes_list[1] + 1: indexes_list[2]]
        sub_seq_2 = solution[indexes_list[2]: indexes_list[3] + 1]
        post_seq = solution[indexes_list[3] + 1:]

        perturbed_solution = prev_seq + sub_seq_2 + mid_seq + sub_seq_1 + post_seq

        perturbed_solution_makespan = Heuristic.Subsequence(
            perturbed_solution,
            self.data.processing_times,
            self.data.setup_times,
            self.data.release_dates
        ).makespan()

        self.TIME_perturbation = self.TIME_perturbation + time.time()
        return perturbed_solution_makespan, perturbed_solution

    def ILS_BS(self, I_R, I_ILS, omega, N, gamma):

        self.TIME_ils_bs = self.TIME_ils_bs - time.time()

        star_solution_makespan = float("inf")
        star_solution = None

        for _ in range(I_R):

            print(f"Starting iteration {_ + 1} of {I_R}")

            current_solution_makespan, current_solution = beam_search(
                omega, N, gamma, self.data.n_jobs, self.data.release_dates, self.data.setup_times,
                self.data.processing_times
            )

            prime_solution = current_solution
            prime_solution_makespan = current_solution_makespan

            IterILS = 0

            while IterILS < I_ILS:

                current_solution_makespan, current_solution = self.local_search(current_solution)

                if current_solution_makespan < prime_solution_makespan:
                    prime_solution = current_solution
                    prime_solution_makespan = current_solution_makespan
                    IterILS = 0

                current_solution_makespan, current_solution = self.Perturbation(prime_solution)
                IterILS = IterILS + 1

            if prime_solution_makespan < star_solution_makespan:
                star_solution = prime_solution
                star_solution_makespan = prime_solution_makespan

            # Print all times
            # print(f" TIME_bs: {self.TIME_bs / 60}m")
            # print(f" TIME_swap: {self.TIME_swap / 60}m")
            # print(f" TIME_l_block: {self.TIME_l_block / 60}m")
            # print(f" TIME_perturbation: {self.TIME_perturbation / 60}m")

        self.TIME_ils_bs = self.TIME_ils_bs + time.time()
        # print(f" TIME_ils_bs: {self.TIME_ils_bs / 60}m")

        return star_solution_makespan, star_solution


if __name__ == "__main__":
    I_R = 2
    I_ILS = 100
    omega = 2
    N = 3
    gamma = 0.5

    instance = "04n_05R"
    print(f"Starting {instance}...")
    heuristic_5 = Heuristic(instance)
    sol_5_makespan, sol_5 = heuristic_5.ILS_BS(I_R, I_ILS, omega, N, gamma)
    print("50n_05R makespan: ", sol_5_makespan)
    print("50n_05R solution: ", sol_5)
