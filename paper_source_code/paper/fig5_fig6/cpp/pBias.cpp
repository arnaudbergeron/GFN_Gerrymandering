// Header files
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;

// [[Rcpp::export]]
Rcpp::NumericVector pBias(Rcpp::NumericVector ndem,
  		  Rcpp::NumericVector nrep,
			  Rcpp::NumericMatrix cdMat,
			  double swing) {

  // Convert swing object
  double swingpct = std::abs(swing);
  
  // Columns of votes with swing
  Rcpp::NumericVector dems(ndem.size());
  Rcpp::NumericVector reps(nrep.size());
  
  // Output vector
  Rcpp::NumericVector rSeats(cdMat.ncol());
  
  // Declare other objects
  int c; Rcpp::NumericVector cds; arma::vec cdVec;
  int i; int ssize; arma::vec cdLabs;
  int nSeats; int dvotes; int rvotes;

  // Loop over columns of cd assignments
  for(int c = 0; c < cdMat.ncol(); c++){
    
    // Get c'th col of cdmat
    cds = cdMat(Rcpp::_,c);

    // Convert to arma objects
    cdVec = Rcpp::as<arma::vec> (cds);
    
    // Get vector of unique cd labels
    cdLabs = arma::unique(cdVec);
    
    // Induce partisan swing
    // Positive is pro-Democrat
    if(swing > 0){
      for(i = 0; i < nrep.size(); i++){
	      ssize = round(nrep(i) * swingpct);
	      reps(i) = nrep(i) - ssize;
	      dems(i) = ndem(i) + ssize;
      }
    }
    if(swing < 0){
      for(i = 0; i < ndem.size(); i++){
	      ssize = round(ndem(i) * swingpct);
	      dems(i) = ndem(i) - ssize;
	      reps(i) = nrep(i) + ssize;
      }
    }
    if(swing == 0){
      dems = ndem;
      reps = nrep;
    }
    
    // Calculate district vote totals  
    nSeats = 0;
    for(i = 0; i < cdLabs.size(); i++){
      
      // Find districts
      arma::uvec precs = find(cdVec == cdLabs(i));
      dvotes = 0;
      rvotes = 0;
      
      // Sum dem, rep votes for that district
      for(int j = 0; j < precs.size(); j++){
	      dvotes += dems(precs(j));
	      rvotes += reps(precs(j));
      }
      
      // If rvotes > dvotes...
      if(rvotes - dvotes > 0){
	      nSeats++;
      }
    }

    rSeats(c) = nSeats;

  }  

  return rSeats;
  
}