from concurrent.futures import ThreadPoolExecutor, as_completed

from mcmc_utils import *
from mcmc_soft import accept_or_reject_proposal, Hyperparameters, A

from dataclasses import dataclass

# Define a dataclass for Hyperparameters
@dataclass
class Hyperparameters:
    q: float
    beta: int
    num_iterations: int
    M: int
    S: int
    lambda_param: float
    max_pop_dev_threshold: float
    compactness_threshold: float
    compactness_constraint: bool

# Define all hyperparameters for each state
ALL_HYPERPARAMS_DICT = {
    "IA": Hyperparameters(0.05, 9, 1000, 10, 5, 2, 0.10, 0.25, True),
    "PA": Hyperparameters(0.04, 20, 200, 10, 5, 10, 0.10, 0.01, False),
    "MA": Hyperparameters(0.10, 30, 100, 10, 5, 2, 0.08, 0.22, True),
    "MI": Hyperparameters(0.10, 30, 100, 10, 5, 5, 0.08, 0.22, False)
}


def run_mcmc_soft(df, q=0.04, beta=30,
                  num_samples=200, lambda_param=2,
                  max_pop_dev_threshold=0.05,
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

    current_partition = initial_partition.copy()
    samples = []
    i = 0

    best_partition = current_partition.copy()
    best_rep_bias = compute_partisan_bias(df, best_partition, dem_vote_col="pre_20_dem_bid",
                                          rep_vote_col="pre_20_rep_tru")
    best_efficiency_gap = compute_efficiency_gap(df, best_partition, dem_vote_col="pre_20_dem_bid",
                                                 rep_vote_col="pre_20_rep_tru")

    total_attempts = 0
    start_time = time.time()

    print("\nStarting sampling...")

    # Helper function for one attempt
    def _attempt_proposal(current_partition):
        # Step 1
        E_on = turn_on_edges(G, current_partition, q)
        # Step 2
        boundary_components = find_boundary_connected_components(G, current_partition, E_on)
        BCP_current_len = len(boundary_components)
        # Step 3
        V_CP, R = select_nonadjacent_components(boundary_components, G, current_partition, lambda_param=lambda_param)
        # Step 4
        proposed_partition = propose_swaps(current_partition, V_CP, G)
        # Step 5: Check constraints
        max_pop_dev = max_population_deviation(proposed_partition, populations)
        avg_compactness = compute_avg_compactness(proposed_partition, gdf)
        if max_pop_dev > max_pop_dev_threshold or avg_compactness < compactness_threshold:
            return None
        # Step 6: Accept or reject
        accepted = accept_or_reject_proposal(current_partition, proposed_partition, G, BCP_current_len, V_CP, R, q,
                                             lambda_param, g_func, df)
        if accepted and proposed_partition != current_partition:
            return (proposed_partition, max_pop_dev, avg_compactness)
        return None

    # Main loop with multithreading for attempts
    while i < num_samples:
        # Instead of one attempt at a time, run multiple attempts in parallel
        # You can tune this batch_size based on how many attempts you want to try simultaneously
        batch_size = 16
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_attempt_proposal, current_partition) for _ in range(batch_size)]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                total_attempts += 1
                if res is not None:
                    # We got a successful partition
                    proposed_partition, max_pop_dev, avg_compactness = res
                    rep_bias = compute_partisan_bias(df, proposed_partition, dem_vote_col="pre_20_dem_bid",
                                                     rep_vote_col="pre_20_rep_tru")
                    efficiency_gap = compute_efficiency_gap(df, proposed_partition,
                                                            dem_vote_col="pre_20_dem_bid",
                                                            rep_vote_col="pre_20_rep_tru")

                    if abs(best_efficiency_gap) >= abs(efficiency_gap) and abs(best_rep_bias) >= abs(
                            rep_bias) and max_pop_dev <= max_pop_dev_threshold * 0.8:
                        best_partition = proposed_partition
                        best_efficiency_gap = efficiency_gap
                        best_rep_bias = rep_bias
                        print(
                            f"Sample {i:03} | Avg Compact: {avg_compactness:.6f} | Max Pop dev: {max_pop_dev * 100:.2f}% | Rep Bias: {rep_bias * 100:.3f}% | Efficiency Gap: {efficiency_gap:.6f} | New Best !"
                        )
                    else:
                        print(
                            f"Sample {i:03} | Avg Compact: {avg_compactness:.6f} | Max Pop dev: {max_pop_dev * 100:.2f}% | Rep Bias: {rep_bias * 100:.3f}% | Efficiency Gap: {efficiency_gap:.6f} "
                        )

                    samples.append(proposed_partition)
                    current_partition = proposed_partition
                    i += 1
                    break  # Stop looking, proceed to next iteration
            else:
                # If we reach here, it means no futures returned a valid result
                # We'll just try again with another batch
                continue

    print("\nEnd of sampling")
    print("Total sampling time:", time.time() - start_time)
    print(
        f"Total attempts: {total_attempts} | Total samples: {len(samples)} | Acceptance Probability: {len(samples) / total_attempts * 100:.2f}%")
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
    - num_samples: Number of iterations to run the algorithm. (more iterations => More samples in output)
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
    state_abrv = "PA"
    seed = 6162
    state_abrv = state_abrv.upper()

    hyperparams_dict = {
        # Small-scale study params for Iowa
        "IA": {
            "q": 0.05,
            "beta": 9,
            "num_samples": 1000,
            "lambda_param": 2,
            "max_pop_dev_threshold": 0.10,
            "compactness_threshold": 0.25
        },
        # Large-scale study optimized params for Pennsylvania
        "PA": {
            "q": 0.05,
            "beta": 30,
            "num_samples": 100,
            "lambda_param": 12,
            "max_pop_dev_threshold": 0.10,
            "compactness_threshold": 0.05,
        },
        "MA": {
            "q": 0.10,
            "beta": 30,
            "num_samples": 100,
            "lambda_param": 2,
            "max_pop_dev_threshold": 0.08,
            "compactness_threshold": 0.22
        },
        "MI": {
            "q": 0.10,
            "beta": 30,
            "num_samples": 100,
            "lambda_param": 5,
            "max_pop_dev_threshold": 0.08,
            "compactness_threshold": 0.22
        }
    }
    q, beta, num_samples, lambda_param, max_pop_dev_threshold, compactness_threshold = hyperparams_dict[
        state_abrv].values()
    random.seed(seed)

    print(f"Running MCMC W/ HARD CONSTRAINTS for {STATE_ABBREVIATIONS[state_abrv]} with the following hyperparameters:")
    print("seed:", seed)
    print("q:", q)
    print("beta:", beta)
    print("num_samples:", num_samples)
    print("lambda_param:", lambda_param)
    print("max_pop_dev_threshold:", max_pop_dev_threshold)
    print("compactness_threshold:", compactness_threshold)

    # Run the algorithm with and tune hyperparameters
    # The best partition is according to an unfixed reward function.
    df = prep_data(state_abrv)
    samples, best_partition = run_mcmc_hard(df,
                                            q=q,
                                            beta=beta,
                                            num_samples=num_samples,
                                            lambda_param=lambda_param,
                                            max_pop_dev_threshold=max_pop_dev_threshold,
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
