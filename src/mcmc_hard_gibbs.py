from mcmc_utils import *


def run_mcmc_hard_gibbs(df, q=0.04, beta=30,
                        num_iterations=200,
                        lambda_param=2,
                        delta=0.05,
                        compactness_threshold=0.22,
                        compactness_constraint=True):
    """
    Algorithm 1.1 (Sampling contiguous redistricting plans with hard constraint)

    Executes the redistricting algorithm for the specified number of samples taken consecutively from the last accepted
    partition. The algorithm follows the steps outlined in the paper while using an updated constraint of geometric
    compactness for each district since the original paper did not include this constraint but is legally required.

    Algorithm 1.1:
    1) Turn on edges in the same district with probability q.
    2) Find boundary components (connected components with neighbors in different districts).
    3) Select a subgroup of nonadjacent components along boundaries that will get swapped.
    4) Propose swaps for the selected components.
    5) Hard check for equal population constraint + geometry compactness.
    6) Accept or reject the proposed partition based on the acceptance probability (Metropolis-Hastings).

    Parameters:
    - df: DataFrame containing the nodes and their attributes.
    - q: Probability threshold for turning on edges.
    - beta: Beta parameter for the Gibbs distribution.
    - num_iterations: Number of iterations to run the algorithm.
    - lambda_param: Lambda parameter for zero-truncated Poisson distribution (used in select_nonadjacent_components).
    - delta: Maximum population deviation allowed for a proposed partition.
    - compactness_threshold: Minimum compactness score allowed for a proposed partition.
    - compactness_constraint: Boolean flag to enable or disable compactness constraint.


    Returns:
    - samples: List of partition samples after each iteration.
    - best_partition: The partition with the highest reward acording to custom reward function.
    """
    # Prepare DataFrame
    df = df.reset_index(drop=True)
    df['node_id'] = df.index

    # Precompute populations and ideal population
    populations = df.set_index('node_id')['pop'].to_dict()
    num_districts = len(set(df['cd_2020']))
    ideal_pop = sum(populations.values()) / num_districts

    # Precompute current partition and create GeoDataFrame
    initial_partition = df.set_index('node_id')['cd_2020'].to_dict()
    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'])

    # Create a NetworkX graph
    G = nx.Graph()
    for i, row in df.iterrows():
        for neighbor in row['adj']:  # row['adj'] should be a list of node ids adjacent to i
            G.add_edge(i, neighbor)

    # Initialize the partition and store initial drawn partitions
    current_partition = initial_partition.copy()

    # Initialize best partition and reward
    best_partition = current_partition.copy()
    best_rep_bias = compute_partisan_bias(df, best_partition,
                                          dem_vote_col="pre_20_dem_bid",
                                          rep_vote_col="pre_20_rep_tru")
    best_efficiency_gap = compute_efficiency_gap(df, best_partition,
                                                 dem_vote_col="pre_20_dem_bid",
                                                 rep_vote_col="pre_20_rep_tru")

    # count iterations
    iteration = 0  # iteration counter (nb time steps actually performed in the chain)
    total_attempts = 0  # total attempts (samples) counter
    start_time = time.time()  # Start time
    samples = []  # List to store partition samples

    print("\nStarting sampling...")

    # Iterative algorithm
    while iteration < num_iterations:
        total_attempts += 1
        # Step 1: Determine E_on edges (Select edges in the same district with probability q)
        E_on = turn_on_edges(G, current_partition, q)

        # Step 2: Find boundary components (connected components with neighbors in different districts)
        boundary_components = find_boundary_connected_components(G, current_partition, E_on)
        BCP_current_len = len(boundary_components)  # precompute |B(CP, π)| for acceptance probability

        # Step 3: Select a subgroup of nonadjacent components along boundaries that will get swapped
        V_CP, R = select_nonadjacent_components(boundary_components, G, current_partition, lambda_param=lambda_param)

        # OPTIONAL: Visualization before changes (optional; can slow down execution by a lot)
        # visualize_map_with_graph_and_geometry(G, E_on, boundary_components, V_CP, df, current_partition)

        # Step 4: Propose swaps for the selected components (random order => can cancel each other out)
        proposed_partition = propose_swaps(current_partition, V_CP, G)

        # Step 5: Hard check for equal population constraint + geometry compactness (and maybe voting share balance later)
        max_pop_dev = max_population_deviation(proposed_partition, populations)
        avg_compactness = compute_avg_compactness(proposed_partition, gdf)
        if compactness_constraint:
            if max_pop_dev >= delta or avg_compactness <= compactness_threshold:
                continue
        else:
            if max_pop_dev >= delta:
                continue

        # Step 6: Accept or reject the proposed partition based on the acceptance probability (Metropolis-Hastings)
        g_current = g_func_cached(current_partition, populations, beta, ideal_pop)
        g_proposed = g_func_cached(current_partition, populations, beta, ideal_pop)
        if proposed_partition != current_partition and accept_or_reject_proposal(current_partition,
                                                                                 proposed_partition,
                                                                                 G, BCP_current_len, V_CP,
                                                                                 R, q, lambda_param,
                                                                                 g_current, g_proposed, df):
            rep_bias = compute_partisan_bias(df, proposed_partition, dem_vote_col="pre_20_dem_bid",
                                             rep_vote_col="pre_20_rep_tru")
            efficiency_gap = compute_efficiency_gap(df, proposed_partition, dem_vote_col="pre_20_dem_bid",
                                                    rep_vote_col="pre_20_rep_tru")
            if abs(best_efficiency_gap) >= abs(efficiency_gap) and abs(best_rep_bias) >= abs(
                    rep_bias) and max_pop_dev <= delta * 0.8:
                best_partition = proposed_partition
                best_efficiency_gap = efficiency_gap
                best_rep_bias = rep_bias
                print(
                    f"Sample {iteration:03} | Avg Compact: {avg_compactness:.6f} | Max Pop dev: {max_pop_dev * 100:.2f}% | "
                    f"Rep Bias: {rep_bias * 100:.3f}% | Efficiency Gap: {efficiency_gap:.6f} | New Best !"
                )
            else:
                print(
                    f"Sample {iteration:03} | Avg Compact: {avg_compactness:.6f} | Max Pop dev: {max_pop_dev * 100:.2f}% | "
                    f"Rep Bias: {rep_bias * 100:.3f}% | Efficiency Gap: {efficiency_gap:.6f} "
                )

            # Save the current partition
            samples.append(proposed_partition)
            current_partition = proposed_partition
            iteration += 1

    print("\nEnd of sampling")
    print("Total sampling time:", time.time() - start_time)
    print(
        f"Total attempts: {total_attempts} | Total samples: {len(samples)} | "
        f"Acceptance Probability: {len(samples) / total_attempts * 100:.2f}%")
    return samples, best_partition


# Define all hyperparameters for each state
ALL_HYPERPARAMS_DICT = {
    "IA": Hyperparameters(0.05, 9, 10000, None, None, 2, 0.07, 0.25, True),
    "PA": Hyperparameters(0.04, 20, 200, None, None, 10, 0.10, 0.05, False),
    "MA": Hyperparameters(0.10, 30, 100, None, None, 2, 0.08, 0.22, True),
    "MI": Hyperparameters(0.10, 30, 100, None, None, 5, 0.08, 0.22, False)
}


def main(state_abrv="IA", seed=6162):
    """
    Main function to run the redistricting algorithm with the specified parameters.

    Parameters (or hyperparameters tuned for Iowa by default, refer to the markdown for more other states):
    - state_abrv: The state abbreviation to load the data for (default: "IA").
    - seed: Random seed for reproducing results (default: 6162).
    - q: Probability threshold for turning on edges in the graph. (higher => More edges turned on, more components)
    - beta: Inverse temperature for the Gibbs distribution. (Higher beta => More equal districts)
    - num_iterations: Number of iterations to run the algorithm. (more iterations => More samples in output)
    - lambda_param: Lambda parameter for zero-truncated Poisson distribution (higher => Change more counties at once)
    - delta: Maximum population deviation from equal districts. (lower => More equal districts)
    - compactness_threshold: Minimum compactness threshold for districts. (higher => More compact districts)

    For smaller states like Iowa, the algorithm can be run with the default hyperparameters. For larger states, the
    hyperparameters may need to be adjusted to achieve better results.

    Outputs:
    - Prints the best reward and partisan metrics for the best partition.
    - Visualizes the best partition on a map.
    - save all the samples to a CSV file.
    """
    state_abrv = state_abrv.upper()
    params = ALL_HYPERPARAMS_DICT[state_abrv]

    # Print hyperparameters
    print(
        f"Running MCMC W/ HARD CONSTRAINTS for {STATE_ABBREVIATIONS[state_abrv]} with the following hyperparameters:\n")
    print(f"seed: {seed}")
    for field, value in params.__dict__.items():
        print(f"{field}: {value}")
    print("\n")

    # Run the algorithm with and tune hyperparameters
    # The best partition is according to an unfixed reward function.
    df = prep_data(state_abrv)
    samples, best_partition = run_mcmc_hard_gibbs(
        df,
        q=params.q,
        beta=params.beta,
        num_iterations=params.num_iterations,
        lambda_param=params.lambda_param,
        delta=params.max_pop_dev_threshold,
        compactness_threshold=params.compactness_threshold,
        compactness_constraint=params.compactness_constraint
    )

    # Just for main:
    # Update the DataFrame with the best partition
    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'])

    # Save partitions to a file
    save_partitions_as_dataframe(samples, output_file=f"output/mcmc_hard_gibbs_partitions_{state_abrv}.csv")

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
        stats.print_stats("mcmc_hard_gibbs.py")
        stats.print_stats("mcmc_utils.py")
        stats.dump_stats("profile_results.pstats")
    else:
        main()
