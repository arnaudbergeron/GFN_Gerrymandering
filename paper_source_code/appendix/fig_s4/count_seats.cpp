// Header files
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;

// Calculate whether one group or another has more votes
// [[Rcpp::export]]
NumericVector count_seats(NumericMatrix partitions, NumericVector g1_pop,
			  NumericVector g2_pop){

  unsigned int m = partitions.ncol(); arma::vec part;
  NumericVector labs = unique(partitions(_,0)); unsigned int j; unsigned int k;
  unsigned int g1_sum; unsigned int g2_sum; int i; arma::uvec assignments;

  NumericVector store_counts(m); unsigned int store_count;
  
  for(i = 0; i < m; i++){

    store_count = 0;

    // Get partition
    part = partitions(_,i);

    for(j = 0; j < labs.size(); j++){

      // Get indices of plans with label j
      assignments = find(part == labs(j));

      // Set sums to 0
      g1_sum = 0; g2_sum = 0;
      
      // Sum vote counts
      for(k = 0; k < assignments.n_elem; k++){

	g1_sum += g1_pop(assignments(k));
	g2_sum += g2_pop(assignments(k));

      }

      // Tally one in store_count if g1 > g2
      if(g1_sum > g2_sum){
	store_count++;
      }

    }

    store_counts(i) = store_count;

  }

  return store_counts;
  
}


