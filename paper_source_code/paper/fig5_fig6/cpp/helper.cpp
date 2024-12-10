// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;

// [[Rcpp::export]]
List genAlConn(List aList, NumericVector cds){

  /* Inputs to function:
     aList: adjacency list of geographic units

     cds: vector of congressional district assignments
  */
  
  // Initialize container list
  List alConnected(cds.size());

  // Initialize
  int i; NumericVector avec; int cd_i; int j;

  // Loop through precincts
  for(i = 0; i < cds.size(); i++){

    // For precinct i, get adjacent precincts
    avec = aList(i);
    
    // Get precinct i's congressional district
    cd_i = cds(i);

    // Initialize empty vector
    NumericVector avec_cd;

    // Loop through avec to identify which are in same cd
    for(j = 0; j < avec.size(); j++){
      
      // Check if j'th entry in avec is same cd, add to avec_cd if so
      if(cds(avec(j)) == cd_i){
	avec_cd.push_back(avec(j));
      }

    }

    // Add to alConnected list
    alConnected(i) = avec_cd;

  }

  return alConnected;

}

// [[Rcpp::export]]
NumericVector findBoundary(List fullList, List conList){

  /* Inputs to function:
     fullList: Full adjacency list of geographic units

     conList: Adjacency list of geographic units within cong district
   */

  // Initialize container vector of 0's (not boundary) and 1's (boundary)
  NumericVector isBoundary(fullList.size());

  // Initialize inside loop
  NumericVector full; NumericVector conn; int i;

  // Loop through aList
  for(i = 0; i < fullList.size(); i++){

    // Get vectors of full and cd-connected components for precinct i
    full = fullList(i);
    conn = conList(i);

    // Compare lengths - if conn < full, then boundary unit
    if(full.size() > conn.size()){
      isBoundary(i) = 1;
    }
    
  }

  return isBoundary;

}

// [[Rcpp::export]]
int countpartitions(List aList) 
{   

  //Takes an adjacency list,
  //The vector of subset nodes
  //The number of subset nodes
						
  //initialize connCompVec   
  //Initialize visited indices
  IntegerVector visitedInd(aList.size());
  int indexVisit = 0;
  
  //Initialize connected components
  IntegerVector currConnComp(aList.size());

  //Initialize the number of connected components
  int numConnComp = 0;
  
  //Loop over nodes
  for(int i = 0; i < aList.size(); i++){
    
    //If i has not been visited...
    if(visitedInd[i] == 0){
      
      //List i as visited
      visitedInd[i] = 1;

      //Increase the number of connected components
      numConnComp++;

      //Add i to the connected component list
      currConnComp[indexVisit] = i;
      
      //increase index visit
      indexVisit++;
      
      //Count the number of nodes in the current connected component
      int nodeCount = indexVisit - 1;
      
      //Initialize a stopping variable:
      int toStop = 0;

      //While we don't stop
      while(toStop == 0){
	
	//get the neighbors of the next current comp
	IntegerVector listNeighs = aList[currConnComp[nodeCount]];
	
	//If listNeighs does not have length zero...
	int listLength = listNeighs.size();
	if(listLength > 0){
	  
	  //Add nodes of listLength to currConnComp
	  //and mark nodes as visited
	  for(int j = 0; j < listLength; j++){
	    if( visitedInd[listNeighs[j]] == 0){
	      currConnComp[indexVisit] = listNeighs[j];
	      visitedInd[listNeighs[j]] = 1;

	      //Increment indexVisit
	      indexVisit++;
	    }
	  }
	}
	
	//Increment nodeCount
	nodeCount++;

	//If currConnComp[nodeCount] is zero, then we must have new connected component
	//Also stop if we have too many guys.
	if(nodeCount == aList.size()){
	  toStop = 1;
	}
	else if(currConnComp[nodeCount] == 0 ){
	  toStop = 1;
	}
      }
    }
  }
  
  return numConnComp;
  
}

