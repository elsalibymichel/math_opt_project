import random
import heapq


# Compute the idle and completion time of a given sequence of jobs
def compute_I_and_C(jobs_sequence, release_dates, setup_times, processing_times):
    # Set the idle time and completion time for the first job
    first_job = jobs_sequence[0]
    idle_time = (
            release_dates[first_job] + setup_times[0][first_job]
    )
    completion_time = idle_time + processing_times[first_job]

    # Iterate over the remaining jobs in the sequence
    for i in range(1, len(jobs_sequence)):
        job = jobs_sequence[i]
        previous_job = jobs_sequence[i - 1]

        # Define the release time, setup time, and processing time for the current job
        release_time_job = release_dates[job]
        setup_time_job = setup_times[previous_job][job] # Setup time from previous job to current job
        processing_time_job = processing_times[job]

        idle_time = max(0, release_time_job - completion_time) + setup_time_job

        completion_time += idle_time + processing_time_job

    return idle_time, completion_time


# Compute a lower bound on the makespan for a given sequence of jobs
# Based on the paper that we are reproducing (formula n. 18 p. 5 of the pdf or 446 of the paper numeration)
def compute_lower_bound(n_jobs, jobs_sequence, release_dates, setup_times, processing_times):
    _, completion_pi = compute_I_and_C(jobs_sequence, release_dates, setup_times, processing_times)

    # Define the set of remaining jobs U(π)
    remaining_jobs_U = [job for job in range(1, n_jobs+1) if job not in jobs_sequence]

    Q = remaining_jobs_U + [jobs_sequence[-1]]

    min_release_time = min(release_dates[j] for j in remaining_jobs_U)

    min_setup_sum = 0
    for k in remaining_jobs_U:
        min_setup_time = min(
            setup_times[t][k]
            for t in Q
            if setup_times[t][k] != -1
        )
        min_setup_sum += min_setup_time

    # Compute sum of processing times
    processing_time_sum = sum(
        processing_times[k] for k in remaining_jobs_U
    )

    lower_bound = (
            max(completion_pi, min_release_time) + min_setup_sum + processing_time_sum
    )

    return lower_bound


# Function β that branches a node π by scheduling job j
def branch_node(j, pi):
    new_pi = pi.copy()
    new_pi.append(j)
    return new_pi


# Stochastic variant of the Beam Search algorithm
# gamma=0 corresponds to the deterministic version of the algorithm
def beam_search(omega, N, gamma, n_jobs, release_dates, setup_times, processing_times):
    # Initialize the root node
    current_level = 0
    pi_0 = []
    current_level_nodes = [pi_0]

    while current_level < n_jobs - 1:
        next_level_nodes = []

        for node in current_level_nodes:
            remaining_jobs_U = [job for job in range(1, n_jobs+1) if job not in node]
            theta = remaining_jobs_U.copy()

            # If the number of jobs exceeds the maximum number of possible branches, remove the job with the
            # maximum idle time
            while len(theta) > omega:
                max_idle_job = max(
                    theta, key=lambda j: compute_I_and_C(node + [j], release_dates, setup_times, processing_times)[0]
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
            lower_bound = compute_lower_bound(n_jobs, node, release_dates, setup_times, processing_times)
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
    # At finale level of the tree keep all branches
    for node in current_level_nodes:
        remaining_jobs_U = [job for job in range(1, n_jobs+1) if job not in node]
        for j in remaining_jobs_U:
            final_nodes.append(branch_node(j, node))

    # Choose the best leaf based on the completion time
    best_leaf = min(final_nodes, key=lambda leaf: compute_I_and_C(leaf, release_dates, setup_times, processing_times)[1])
    _, best_completion_time = compute_I_and_C(best_leaf, release_dates, setup_times, processing_times)

    return best_completion_time, best_leaf
