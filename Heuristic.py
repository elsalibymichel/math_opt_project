from PreProcess import PreProcess

import random
import heapq
import time
import datetime


# Function β that branches a node π by scheduling job j
def branch_node(j, pi):
    new_pi = pi.copy()
    new_pi.append(j)
    return new_pi


class Heuristic:
    def __init__(self, file_path):
        self.data = PreProcess(file_path)
        self.all_jobs = list(range(1, self.data.n_jobs + 1))

        # Some variables to track time
        self.TIME_bs = 0
        self.TIME_swap = 0
        self.TIME_l_block = 0
        self.TIME_perturbation = 0
        self.TIME_ils_bs = 0

    # Idle and completion time
    def compute_I_and_C(self, jobs_sequence):
        # Set the idle time and completion time for the first job
        first_job = jobs_sequence[0]
        idle_time = (
                self.data.release_dates[first_job] + self.data.setup_times[0][first_job]
        )
        completion_time = idle_time + self.data.processing_times[first_job]

        # Iterate over the remaining jobs in the sequence
        for i in range(1, len(jobs_sequence)):
            job = jobs_sequence[i]
            previous_job = jobs_sequence[i - 1]

            # Define the release time, setup time, and processing time for the current job
            release_time_job = self.data.release_dates[job]
            setup_time_job = self.data.setup_times[previous_job][job]
            processing_time_job = self.data.processing_times[job]

            idle_time = max(0, release_time_job - completion_time) + setup_time_job

            completion_time += idle_time + processing_time_job

        return idle_time, completion_time

    # Lower bound
    def compute_lower_bound(self, jobs_sequence, all_jobs):
        _, completion_pi = self.compute_I_and_C(jobs_sequence)

        # Define the set of remaining jobs U(π)
        remaining_jobs_U = [job for job in all_jobs if job not in jobs_sequence]

        Q = remaining_jobs_U + [jobs_sequence[-1]]

        min_release_time = min(self.data.release_dates[j] for j in remaining_jobs_U)

        min_setup_sum = 0
        for k in remaining_jobs_U:
            min_setup_time = min(
                self.data.setup_times[t][k]
                for t in Q
                if self.data.setup_times[t][k] != -1
            )
            min_setup_sum += min_setup_time

        # Compute sum of processing times
        processing_time_sum = sum(
            self.data.processing_times[k] for k in remaining_jobs_U
        )

        lower_bound = (
                max(completion_pi, min_release_time) + min_setup_sum + processing_time_sum
        )

        return lower_bound

    # Algorithm Beam Search
    def beam_search(self, w, N, gamma):

        self.TIME_bs = self.TIME_bs - time.time()

        # Initialize the root node
        current_level = 0
        pi_0 = []
        current_level_nodes = [pi_0]

        while current_level < len(self.all_jobs) - 1:
            next_level_nodes = []

            for node in current_level_nodes:
                remaining_jobs_U = [job for job in self.all_jobs if job not in node]
                theta = remaining_jobs_U.copy()

                # If the number of jobs exceeds the maximum number of possible branches, remove the job with the
                # maximum idle time
                while len(theta) > w:
                    max_idle_job = max(
                        theta, key=lambda j: self.compute_I_and_C(node + [j])[0]
                    )
                    theta.remove(max_idle_job)

                # Branch the node by scheduling each job in theta
                for j in theta:
                    new_node = branch_node(j, node)
                    next_level_nodes.append(new_node)

            candidate_list = []

            # Compute the lower bound for each node in next_level_nodes and add them to the candidate list based on
            # the lower bound Heap queue is used for keeping the candidate list sorted
            for node in next_level_nodes:
                lower_bound = self.compute_lower_bound(node, self.all_jobs)
                heapq.heappush(candidate_list, (lower_bound, node))

            # Randomly select N nodes from the candidate list
            next_level_nodes = [
                heapq.heappop(candidate_list)[1]
                for _ in range(min(len(candidate_list), int((1 + gamma) * N)))
            ]
            random.shuffle(next_level_nodes)
            next_level_nodes = next_level_nodes[:N]

            # Update the list of the sequence of nodes
            current_level_nodes = next_level_nodes
            current_level += 1

        final_nodes = []
        # Finale level of the tree
        for node in current_level_nodes:
            remaining_jobs_U = [job for job in self.all_jobs if job not in node]
            for j in remaining_jobs_U:
                final_nodes.append(branch_node(j, node))

        best_node = min(final_nodes, key=lambda node: self.compute_I_and_C(node)[1])
        _, best_completion_time = self.compute_I_and_C(best_node)

        self.TIME_bs = self.TIME_bs + time.time()

        return best_completion_time, best_node

    # Subsequence class
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

            if len(jobs) == 1:
                if jobs[0] == 0:
                    self.init_dummy_sequence()
                    return
                else:
                    self.private_init_one_job(jobs[0])
                    return
            else:
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

        def dummy_sequence(self):
            return Heuristic.Subsequence(
                [0], self.processing_times, self.setup_times, self.release_dates
            )

        def init_dummy_sequence(self):
            self.F = 0
            self.L = 0
            self.E = 0
            self.D = 0
            return

        def private_init_one_job(self, job):
            self.F = job
            self.L = job
            self.E = self.release_dates[job]
            self.D = self.processing_times[job]
            return

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

        def C(self):
            return self.E + self.D

        def makespan(self):
            if len(self.jobs) != len(self.release_dates) - 1:
                raise Exception("Seqence does not contain all jobs")
            else:
                dummy = self.dummy_sequence()
                complete_solution = dummy.concatenate_to(self)
                makespan = complete_solution.C()
                return makespan
                # return self.dummy_sequence().concatenate_to(self).C()

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

    # Solution is a list of jobs
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

            current_solution_makespan, current_solution = self.beam_search(omega, N, gamma)

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

    def generate_instance(self):
        pass


if __name__ == "__main__":
    I_R = 10
    I_ILS = 100
    omega = 2
    N = 3
    gamma = 0.5

    instance = "20n_05R"
    print(f"Starting {instance}...")
    heuristic_5 = Heuristic(instance)
    sol_5_makespan, sol_5 = heuristic_5.ILS_BS(I_R, I_ILS, omega, N, gamma)
    print("50n_05R makespan: ", sol_5_makespan)

# if __name__ == "__main__":
#     # n_jobs = 5
#     # processing_times = [-1, 1, 2, 1, 2, 2]
#     # release_dates = [-1, 1, 3, 10, 2, 4]
#     # setup_times = [
#     #     [-1, 2, 5, 1, 5, 7],
#     #     [-1, -1, 9, 8, 1, 5],
#     #     [-1, 4, -1, 6, 5, 3],
#     #     [-1, 1, 3, -1, 8, 2],
#     #     [-1, 10, 2, 3, -1, 7],
#     #     [-1, 8, 1, 5, 7, -1],
#     # ]
#     #
#     # initial_solution = [3, 4, 1, 2, 5]
#
#     data = PreProcess("Instances/in02_001.dat")
#     n_jobs = data.n_jobs
#     processing_times_1 = data.processing_times
#     release_dates_1 = data.release_dates
#     setup_times_1 = data.setup_times
#
#     heuristic = Heuristic("Instances/in02_001.dat")
#
#     best_makespan, best_solution = heuristic.ILS_BS(I_R=10, I_ILS=100, omega=2, N=3, gamma=0.5)
#     print(best_makespan, best_solution)
#     # initial_solution = heuristic.beam_search(2, 3, 0.5)
#
#     # subseq0 = Heuristic.Subsequence(
#     #     [1, 4, 3], processing_times, setup_times, release_dates
#     # )
#     # subseq1 = Heuristic.Subsequence(
#     #     [5, 2], processing_times, setup_times, release_dates
#     # )
#
#     # if subseq0.concatenate_to(subseq1).makespan() == 21:
#     #     print("Subsequence class: OK")
#     # else:
#     #     print("Subsequence class: WRONG")
#
#     # test_local_search = Heuristic.Local_Search(
#     #     inital_solution, processing_times, setup_times, release_dates
#     # )
#
#     # for l in range(1, 5):
#     #     print(f"{l}-block", test_local_search.get_best_from_l_block_neighborhood(l, inital_solution))
#
#     # print("swap", test_local_search.get_best_from_swap_neighborhood(inital_solution))
#
#     # print("Now perform local search algorithm:")
#     # local_search_solution = test_local_search.local_search()
#     # local_search_solution_makespan = Heuristic.Subsequence(local_search_solution,
#     #         processing_times,
#     #         setup_times,
#     #         release_dates
#     # ).makespan()
#     # print(local_search_solution_makespan, local_search_solution)
