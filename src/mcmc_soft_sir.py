from mcmc_utils import *
from mcmc_soft import accept_or_reject_proposal


def run_mcmc_soft_with_sir(df, q=0.04, beta=30,
                           num_iterations=20,  # Number of outer iterations
                           M=10,  # M samples generated each iteration
                           S=5,  # S samples drawn from SIR
                           lambda_param=2,
                           delta=0.05,
                           compactness_threshold=0.22):
    """
    Closer implementation to the algorithm described in the image of Algorithm 1.2 (soft constraint).

    Steps:
    - We have num_iterations "cycles".
    - In each iteration:
      1) Run the basic algorithm to generate M samples using MH acceptance probability with Gibbs.
      2) Reject samples that fail the population constraint (max population deviation > delta).
      3) From the remaining, perform SIR resampling with weights = 1/g_beta(π) to get S samples.
      4) Append these S samples to the global samples list.

    Arguments:
    - df: DataFrame with nodes and their attributes.
    - reward_w: dictionary of weights for the reward function.
    - q, beta, lambda_param: parameters as in the original code.
    - num_iterations: how many times we run the M-sample generation + filtering + SIR cycle.
    - M: how many samples we generate each iteration.
    - S: how many samples we draw after SIR.
    - delta: population constraint threshold.

    Returns:
    - samples: list of all final samples (S samples per iteration * num_iterations)
    - best_partition: best partition found so far.
    """
    # Precompute node-level data for reward calculations and prepare DataFrame
    df = df.reset_index(drop=True)
    df['node_id'] = df.index
    populations = df.set_index('node_id')['pop'].to_dict()
    initial_partition = df.set_index('node_id')['cd_2020'].to_dict()
    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'])

    # Create a NetworkX graph
    G = nx.Graph()
    for i, row in df.iterrows():
        for neighbor in row['adj']:  # row['adj'] should be a list of node ids adjacent to i
            G.add_edge(i, neighbor)

        # Define g_func here so it can access `populations` and `beta` efficiently for each computation
        def g_func(partition):
            """
            For every acceptance probability calculation, we need to compute this for current and proposed partitions.
            Compute g(π) the Gibbs distribution unnormalized probability:
            g(π) = exp(-β * sum_over_districts |(pop(district)/ideal_pop - 1)|)

            The more the populations among districts deviate from the ideal population, the lower the probability.

            In the paper, they make this more efficient by computing the differences in populations.
            """
            # Aggregate district populations using defaultdict
            district_pop = defaultdict(float)
            for node, dist in partition.items():
                district_pop[dist] += populations[node]

            # Precompute constants
            total_pop = sum(populations.values())
            num_districts = len(district_pop)
            ideal_pop = total_pop / num_districts

            # Compute deviation sum
            deviation_sum = sum(abs((pop / ideal_pop) - 1) for pop in district_pop.values())

            # Return g(π), the Gibbs distribution unnormalized probability
            return np.exp(-beta * deviation_sum)

    # Initialize
    drawn_partitions = [initial_partition.copy()]  # Start from initial partition
    # Initialize the partition and store initial results for visualization
    current_partition = initial_partition.copy()

    # List to store partition samples
    samples = []
    iteration = 0

    # Initialize best partition and reward
    best_partition = current_partition.copy()
    best_rep_bias = compute_partisan_bias(df, best_partition, dem_vote_col="pre_20_dem_bid",
                                          rep_vote_col="pre_20_rep_tru")
    best_efficiency_gap = compute_efficiency_gap(df, best_partition, dem_vote_col="pre_20_dem_bid",
                                                 rep_vote_col="pre_20_rep_tru")

    # count iterations
    total_attempts = 0
    start_time = time.time()

    print("Starting Algorithm 1.2 (Soft constraint) ...")

    while iteration < num_iterations:
        # Step 1: Generate M samples using the basic algorithm.
        # Important: Do NOT filter by population/compactness constraint here, only use the acceptance probability.
        iteration_samples = []
        attempts = 0
        while len(iteration_samples) < M:
            attempts += 1
            # Start from a partition in drawn_partitions
            current_partition = random.choice(drawn_partitions)

            # Perform steps as in the basic algorithm:
            E_on = turn_on_edges(G, current_partition, q)  # 1. Turn on edges
            boundary_components = find_boundary_connected_components(G, current_partition, E_on)  # 2. Find Boundary CPs
            BCP_current_len = len(boundary_components)  # Store for later use
            V_CP, R = select_nonadjacent_components(boundary_components, G, current_partition,
                                                    lambda_param=lambda_param)  # 3. Select V_CP
            proposed_partition = propose_swaps(current_partition, V_CP, G)

            # Note: Here we do NOT check population or compactness yet.
            # We just check acceptance probability.
            accepted = accept_or_reject_proposal(current_partition,
                                                 proposed_partition,
                                                 G, BCP_current_len, V_CP, R, q,
                                                 lambda_param, g_func, df)
            if accepted and proposed_partition != current_partition:
                iteration_samples.append(proposed_partition.copy())

        # Now we have M samples accepted by the MH step.

        # Step 2: Reject samples failing the population constraint.
        # The paper states "For a specified population constraint δ, discard those samples."
        # Here we interpret the constraint: max_{ℓ} |(Sum_i∈Vℓ p_i)/p̃ - 1| > δ
        # This is basically the same as max_pop_dev > δ.
        valid_samples = []
        for samp in iteration_samples:
            max_dev = max_population_deviation(samp, populations)
            avg_compactness = compute_avg_compactness(samp, gdf)
            if max_dev <= delta and avg_compactness >= compactness_threshold:
                # Keep only samples that meet the population constraint
                valid_samples.append(samp)

        if len(valid_samples) == 0:
            print(f"Iteration {iteration:02}: No valid samples after applying population constraint. Continuing...")
            # If no valid samples, we might just continue with next iteration
            # or handle differently. For now, let's just continue.
            # No SIR is performed.
            continue

        # Step 3: SIR resampling from the valid samples using weights = 1/g_β(π)
        weights = np.array([1 / g_func(samp) for samp in valid_samples])
        weights /= weights.sum()

        # Draw S samples with replacement
        chosen_indices = np.random.choice(len(valid_samples), size=S, p=weights, replace=True)
        drawn_partitions = [valid_samples[idx].copy() for idx in chosen_indices]

        # Compute metrics for all drawn partitions
        rep_biases = [compute_partisan_bias(df, p, dem_vote_col="pre_20_dem_bid", rep_vote_col="pre_20_rep_tru")
                      for p in drawn_partitions]
        efficiency_gaps = [compute_efficiency_gap(df, p, dem_vote_col="pre_20_dem_bid", rep_vote_col="pre_20_rep_tru")
                           for p in drawn_partitions]
        max_pop_devs = [max_population_deviation(p, populations) for p in drawn_partitions]
        avg_compactnesses = [compute_compactness(gdf, p)[0] for p in drawn_partitions]

        # Find the best partition based on efficiency gap, rep bias, and population deviation
        valid_best_indices = [i for i, (eg, rb, mpd) in enumerate(zip(efficiency_gaps, rep_biases, max_pop_devs))
                              if abs(eg) <= abs(best_efficiency_gap)
                              and abs(rb) <= abs(best_rep_bias)
                              and mpd <= delta * 0.8]

        if valid_best_indices:
            # Select the best partition based on criteria
            best_idx = min(valid_best_indices, key=lambda i: (abs(efficiency_gaps[i]), abs(rep_biases[i])))
            best_partition = drawn_partitions[best_idx].copy()
            best_efficiency_gap = efficiency_gaps[best_idx]
            best_rep_bias = rep_biases[best_idx]
            print(
                f"\nNew Best! | Avg Compact: {avg_compactnesses[best_idx]:.6f} | "
                f"Max Pop dev: {max_pop_devs[best_idx] * 100:.2f}% | "
                f"Rep Bias: {rep_biases[best_idx]:.3f} | Efficiency Gap: {efficiency_gaps[best_idx]:.6f}\n"
            )
        # Append these S samples to global samples
        samples.extend(drawn_partitions)

        # Print iteration info
        avg_compactness = np.mean(avg_compactnesses)
        avg_pop_dev = np.mean(max_pop_devs)
        avg_efficiency_gap = np.mean(efficiency_gaps)
        avg_rep_bias = np.mean(rep_biases)
        print(
            f"Iteration {iteration:02} | "
            f"Avg Compact: {avg_compactness:.6f} | Avg Pop Dev: {avg_pop_dev:.6f} | "
            f"Avg Eff. Gap: {avg_efficiency_gap:.6f} | Avg Rep Bias: {avg_rep_bias:.3f}"
        )
        total_attempts += attempts
        iteration += 1

    print("End of Algorithm 1.2 (Soft constraint)")
    print("Total sampling time:", time.time() - start_time)
    print(
        f"Total attempts: {total_attempts} | Total samples: {len(samples)} | "
        f"Acceptance Probability: {len(samples) / total_attempts * 100:.2f}%")
    return samples, best_partition


def main(state_abrv="PA", seed=6162):
    """
    Main function to run the redistricting algorithm with the specified parameters.

    These are all the parameters possible to tune for the algorithm. The hyperparameters are tuned for the state of
    Iowa (IA) by default.

    Parameters (or hyperparameters tuned for Iowa by default, refer to the markdown for more other states):
    - state_abrv: The state abbreviation to load the data for (default: "IA").
    - seed: Random seed for reproducing results (default: 6162).
    - q: Probability threshold for turning on edges in the graph. (higher => More edges turned on, more components)
    - beta: Inverse temperature for the Gibbs distribution. (Higher beta => More equal districts)
    - num_iterations: Number of iterations to run the algorithm. (more iterations => More samples in output)
    - lambda_param: Lambda parameter for zero-truncated Poisson distribution (higher => Change more counties at once)
    - max_pop_dev_threshold: Maximum population deviation from equal districts. (lower => More equal districts)
    - compactness_threshold: Minimum compactness threshold for districts. (higher => More compact districts)

    For smaller states like Iowa, the algorithm can be run with the default hyperparameters. For larger states, the
    hyperparameters may need to be adjusted to achieve better results.

    Outputs:
    - Prints the best reward and partisan metrics for the best partition.
    - Visualizes the best partition on a map.
    - save all the samples to a CSV file.
    """
    state_abrv = state_abrv.upper()
    hyperparams_dict = {
        # Small-scale study params for Iowa
        "IA": {
            "q": 0.05,
            "beta": 9,
            "num_iterations": 1000,
            "M": 10,
            "S": 5,
            "lambda_param": 2,
            "max_pop_dev_threshold": 0.075,
            "compactness_threshold": 0.25
        },
        # Large-scale study optimized params for Pennsylvania
        "PA": {
            "q": 0.04,
            "beta": 20,
            "num_iterations": 100,
            "M": 10,
            "S": 5,
            "lambda_param": 10,
            "max_pop_dev_threshold": 0.10,
            "compactness_threshold": 0.10,
        },
        "MA": {
            "q": 0.10,
            "beta": 30,
            "num_iterations": 100,
            "M": 10,
            "S": 5,
            "lambda_param": 2,
            "max_pop_dev_threshold": 0.08,
            "compactness_threshold": 0.22
        },
        "MI": {
            "q": 0.10,
            "beta": 30,
            "num_iterations": 100,
            "M": 10,
            "S": 5,
            "lambda_param": 5,
            "max_pop_dev_threshold": 0.08,
            "compactness_threshold": 0.22
        }
    }
    q, beta, num_iterations, M, S, lambda_param, max_pop_dev_threshold, compactness_threshold = hyperparams_dict[
        state_abrv].values()
    random.seed(seed)

    print(f"Running MCMC W/ HARD CONSTRAINTS for {STATE_ABBREVIATIONS[state_abrv]} with the following hyperparameters:")
    print("seed:", seed)
    print("q:", q)
    print("beta:", beta)
    print("num_iterations:", num_iterations)
    print("M:", M)
    print("S:", S)
    print("lambda_param:", lambda_param)
    print("max_pop_dev_threshold:", max_pop_dev_threshold)
    print("compactness_threshold:", compactness_threshold)

    # Run the algorithm with and tune hyperparameters
    # The best partition is according to an unfixed reward function.
    df = prep_data(state_abrv)
    samples, best_partition = run_mcmc_soft_with_sir(df,
                                                     q=q,
                                                     beta=beta,
                                                     num_iterations=num_iterations,
                                                     M=M,
                                                     S=S,
                                                     lambda_param=lambda_param,
                                                     delta=max_pop_dev_threshold,
                                                     compactness_threshold=compactness_threshold)

    # Just for main:
    # Update the DataFrame with the best partition
    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'])

    # Save partitions to a file
    save_partitions_as_dataframe(samples, output_file=f"output/mcmc_soft_partitions_{state_abrv}.csv")

    ##############################################################
    # Best run: calculating metrics for the best partition
    df = df.reset_index(drop=True)
    df['node_id'] = df.index
    best_district = np.array([best_partition[node] for node in range(len(df))])
    df['district'] = best_district
    populations = df.set_index('node_id')['pop'].to_dict()

    # Precincts changed
    if best_partition == df.set_index('node_id')['cd_2020'].to_dict():
        print("NO CHANGES made to the initial partition.")
    else:
        # Print the difference between the initial and final partition in pourcentage
        changed_precincts = calculate_percentage_changed(best_partition, df.set_index('node_id')['cd_2020'].to_dict())
        print(f"Percentage of precincts changed: {changed_precincts:.2f}%")

    rep_bias = compute_partisan_bias(df, best_partition, dem_vote_col="pre_20_dem_bid", rep_vote_col="pre_20_rep_tru")
    efficiency_gap = compute_efficiency_gap(df, best_partition, dem_vote_col="pre_20_dem_bid",
                                            rep_vote_col="pre_20_rep_tru")
    compactness_mean, compactness_std = compute_compactness(gdf, best_partition)
    pop_variance = compute_population_entropy(df, best_partition)
    # Print the partisan metrics for the best partition
    print()
    print("Republican bias:", round(rep_bias, 6))
    print("Efficiency gap:", round(efficiency_gap, 6))
    print("Compactness Mean:", round(compactness_mean, 6))
    print("Compactness std:", round(compactness_std, 6))
    print("Population entropy:", round(pop_variance, 6))
    print("Max population deviation:", round(max_population_deviation(best_partition, populations), 6))

    # Visualize the best partition
    metrics = {
        "total": [("pop", "Total Population"), ("vap", "Voting Age Population")],
        "mean": [],
        "ratio": [[("pre_20_dem_bid", "Biden"), ("pre_20_rep_tru", "Trump")]]
    }
    visualize_map_with_geometry(df,
                                geometry_col="geometry", district_id_col=best_partition,
                                state=STATE_ABBREVIATIONS[state_abrv], metrics=metrics)


if __name__ == "__main__":
    profile_run = True
    if profile_run:
        profiler = cProfile.Profile()
        profiler.runcall(main)
        stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
        stats.print_stats("mcmc_hard.py")
        stats.print_stats("data_utils.py")
        stats.dump_stats("profile_results.pstats")
    else:
        main()
