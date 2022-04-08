
# -------------------------------------------------------------------------
#  Hierarchical Hidden Markov Process with State Prediction: Misc. functions
#  Author: Kevin L. McKee (klmckee@ucdavis.edu)
#  Description: This script contains miscellaneous functions for use in simulation, data analysis, plotting, etc.
# -------------------------------------------------------------------------

#Used internally to place parameters from a vector into a 2D transition matrix.
genIndMat<-function(K){ 
  m<-matrix(0, K, K)
  ind<-1
  for(j in 1:K) for(i in 1:K){
    if(i!=j){
      m[i,j]<-ind
      ind<-ind+1
    }
  } 
  return(m)
}

# Modifying pairs plot to mark the zero points of each axis
panelPlot<-function(x, y, ...) {
  points(x,y, pch=".", cex=2)
  abline(h=0)
  abline(v=0)
}
