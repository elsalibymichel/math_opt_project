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
        self.TIME_ils_bs = 0  # This is also the duration of the heuristic

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
        # This method is used for efficiency reasons
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

    # Return the best neighbor, ie, a couple (makespan, neighbor), of a solution in the swap neighborhood
    # The swap neighborhood is defined by all possible swaps of two jobs in the sequence
    def get_best_from_swap_neighborhood(self, solution: list):

        # Track time spent in this function
        self.TIME_swap = self.TIME_swap - time.time()

        best_neighbor = None
        best_makespan = float("inf")
        for i in range(len(solution) - 2):
            jobs_prev_i = solution[:i]
            if i != 0:
                # For efficient move evaluation, pre-compute the subsequence attributes of the jobs
                # before the i-th index
                sub_seq_prev_i = Heuristic.Subsequence(
                    jobs_prev_i,
                    self.data.processing_times,
                    self.data.setup_times,
                    self.data.release_dates,
                )

            for j in range(i + 1, len(solution) - 1):
                jobs = solution.copy()
                # Swap jobs in index position i and j
                jobs[i], jobs[j] = jobs[j], jobs[i]
                sub_seq_post_i = Heuristic.Subsequence(
                    jobs[i:],
                    self.data.processing_times,
                    self.data.setup_times,
                    self.data.release_dates,
                )

                # Check if the current neighbor is better than the best neighbor found so far
                makespan = None
                if i == 0:
                    makespan = sub_seq_post_i.makespan()
                else:
                    makespan = sub_seq_prev_i.concatenate_to(sub_seq_post_i).makespan()

                if makespan < best_makespan:
                    best_makespan = makespan
                    best_neighbor = jobs

        # Track time spent in this function
        self.TIME_swap = self.TIME_swap + time.time()

        return best_makespan, best_neighbor

    # Return the best neighbor, ie, a couple (makespan, neighbor), of a solution in the l-block neighborhood
    # The l-block neighborhood is defined by shifting all possible blocks of l jobs in all possible indexes
    def get_best_from_l_block_neighborhood(self, l, solution: list):

        # Track time spent in this function
        self.TIME_l_block = self.TIME_l_block - time.time()

        best_neighbor = None
        best_neighbor_makespan = float("inf")
        for i in range(len(solution) - l + 1):
            # For efficient move evaluation, pre-compute the subsequence attributes of the jobs
            # between i and i+l indexes
            sub_seq_i_l = Heuristic.Subsequence(
                solution[i: i + l],
                self.data.processing_times,
                self.data.setup_times,
                self.data.release_dates,
            )

            # Find the feasible indexes to place the l-block under evaluation
            legit_indexes = [
                k for k in range(len(solution)) if k not in range(i, i + l)
            ]

            for j in legit_indexes:

                # Split the solution in three parts: prev_block, block, post_block
                # This depends on the position of the l-block in the solution
                if j >= i + l:
                    prev_block = solution[:i] + solution[i + l: j + 1]
                    block = solution[i: i + l]
                    post_block = solution[j + 1:]
                else:
                    prev_block = solution[:j]
                    block = solution[i: i + l]
                    post_block = solution[j:i] + solution[i + l:]

                # The neighbor is defined by (prev_block + block + post_block)
                # But, for using efficient move evaluation, we have to check if these subsequences are empty
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
                    else:
                        makespan = sub_seq_prev.concatenate_to(sub_seq_i_l).makespan()
                else:
                    if post_block:
                        sub_seq_post = Heuristic.Subsequence(
                            post_block,
                            self.data.processing_times,
                            self.data.setup_times,
                            self.data.release_dates,
                        )
                        makespan = sub_seq_i_l.concatenate_to(sub_seq_post).makespan()
                        # print(3, block, post_block)
                    else:
                        raise Exception("l-block should not be l=n_jobs")

                # Check if the current neighbor is better than the best neighbor found so far
                if makespan < best_neighbor_makespan:
                    best_neighbor_makespan = makespan
                    best_neighbor = prev_block + block + post_block

        # Track time spent in this function
        self.TIME_l_block = self.TIME_l_block + time.time()

        return best_neighbor_makespan, best_neighbor

    # Perform the local search algorithm on a given solution described by A. Subramanian et al.
    #   "A parallel heuristic for the Vehicle Routing Problem with Simultaneous Pickup and Delivery"
    #   (2010) Computers & Operations Research
    def local_search(self, solution):

        # In the neighborhood_list, the index k=0 is related to the swap neighborhood
        # while the indexes k=l (l!=0) are related to the l-block neighborhood (considering all possible values of l)
        neighborhood_list = [k for k in range(self.data.n_jobs)]

        # Initialize the current best solution and its makespan
        current_best_solution = solution
        current_best_makespan = Heuristic.Subsequence(
            solution, self.data.processing_times, self.data.setup_times, self.data.release_dates
        ).makespan()

        while neighborhood_list:

            neighborhood = random.choice(neighborhood_list)

            # Get the best neighbor from the chosen neighborhood
            new_solution_makespan = None
            if neighborhood == 0:  # Swap
                new_solution_makespan, new_solution = self.get_best_from_swap_neighborhood(current_best_solution)
            else:  # l-block
                l = neighborhood
                new_solution_makespan, new_solution = self.get_best_from_l_block_neighborhood(l, current_best_solution)

            # Check if the current neighbor is better than the best neighbor found so far
            if new_solution_makespan < current_best_makespan:
                current_best_solution = new_solution
                current_best_makespan = new_solution_makespan

                # neighborhood_list is restored with all possible neighborhoods
                neighborhood_list = [k for k in range(self.data.n_jobs)]
            else:
                # Remove the chosen neighborhood from the list
                neighborhood_list.remove(neighborhood)

        return current_best_makespan, current_best_solution

    # Perform the perturbation operator on a given solution
    # Given a solution, exchange the positions of two disjoint subsequences selected randomly
    def Perturbation(self, solution):

        # Track time spent in this function
        self.TIME_perturbation = self.TIME_perturbation - time.time()

        # Select four indexes randomly
        indexes_list = random.sample([k for k in range(0, self.data.n_jobs)], 4)
        indexes_list.sort()

        # Divide the sequence in five parts: prev_seq, sub_seq_1, mid_seq, sub_seq_2, post_seq
        prev_seq = solution[: indexes_list[0]]
        sub_seq_1 = solution[indexes_list[0]: indexes_list[1] + 1]
        mid_seq = solution[indexes_list[1] + 1: indexes_list[2]]
        sub_seq_2 = solution[indexes_list[2]: indexes_list[3] + 1]
        post_seq = solution[indexes_list[3] + 1:]

        # Concatenate the subsequences in a new solution exchanging the positions of sub_seq_1 and sub_seq_2
        perturbed_solution = prev_seq + sub_seq_2 + mid_seq + sub_seq_1 + post_seq

        # Compute the makespan of the perturbed solution
        perturbed_solution_makespan = Heuristic.Subsequence(
            perturbed_solution,
            self.data.processing_times,
            self.data.setup_times,
            self.data.release_dates
        ).makespan()

        # Track time spent in this function
        self.TIME_perturbation = self.TIME_perturbation + time.time()

        return perturbed_solution_makespan, perturbed_solution

    # Perform the ILS-BS algorithm
    def ILS_BS(self, I_R, I_ILS, omega, N, gamma):

        # Track time spent in this function
        self.TIME_ils_bs = self.TIME_ils_bs - time.time()

        # Initialize the best solution found so far (global best solution)
        star_solution_makespan = float("inf")
        star_solution = None

        for _ in range(I_R):

            print(f"Starting iteration {_ + 1} of {I_R}")

            # Perform the Beam Search algorithm
            current_solution_makespan, current_solution = beam_search(
                omega, N, gamma, self.data.n_jobs, self.data.release_dates, self.data.setup_times,
                self.data.processing_times
            )

            # Initialize the best solution of the current I_R iteration
            prime_solution = current_solution
            prime_solution_makespan = current_solution_makespan

            IterILS = 0

            while IterILS < I_ILS:

                # Perform the Local Search algorithm
                current_solution_makespan, current_solution = self.local_search(current_solution)

                # Check if the current solution is better than the best solution of the current I_R iteration
                if current_solution_makespan < prime_solution_makespan:
                    prime_solution = current_solution
                    prime_solution_makespan = current_solution_makespan
                    IterILS = 0

                current_solution_makespan, current_solution = self.Perturbation(prime_solution)
                IterILS = IterILS + 1

            # Check if the best solution of the current I_R iteration is better than the global best solution
            if prime_solution_makespan < star_solution_makespan:
                star_solution = prime_solution
                star_solution_makespan = prime_solution_makespan

        # Track time spent in this function
        self.TIME_ils_bs = self.TIME_ils_bs + time.time()

        return star_solution_makespan, star_solution


# Testing
if __name__ == "__main__":
    I_R = 2
    I_ILS = 100
    omega = 2
    N = 3
    gamma = 0.5

    instance = "experiments_results/30n_15R_4"
    print(f"Starting {instance}...")
    heuristic_5 = Heuristic(instance)
    sol_5_makespan, sol_5 = heuristic_5.ILS_BS(I_R, I_ILS, omega, N, gamma)
    print(sol_5_makespan)
    print("Duration: ", heuristic_5.TIME_ils_bs)
