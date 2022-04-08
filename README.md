# Hidden Markov Models Workshop scripts
This is code for simulating data from Hidden Markov Models (HMM) and fitting HMMs in Stan.
Currently, the most annotated scripts are model3_HMM_dataAnalysis and model3_HMM_dataSimulation, so for a practicalwalk-through, it is easiest to start with those. 

- model1 is the basic HMM. Transition probabilities are estimated for a set of simple logistic regression sub-models.
- model2 introduces state prediction, such that the most likely sub-model can be conditioned on the covariates.
- model3 introduces hierarchical parameter estimation such that many individuals can be modeled as a group. Parameter means, standard deviations, and individual scores are estimated.
- model3_test allows inclusion of a separate test data set for model comparisons.

The data analysis script uses model3_test to compare models with 2, 3, and 4 hidden states using out-of-sample likelihood.
Sampling over several random chains is used to deal with the label-switching problem and to produce samples of the test likelihood.
