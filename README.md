# NEAT-regularization
Reduce the overfitting problem on neural evolution by regularising the network topology complexity

This is the source code of my research project.
Each fold in this project contains one dataset and all the source code of this dataset.

preprocessing.ipynb is used to preprocess the data and the preprocessed data will be stored in: train_feature_pickle, train_target_pickle, test_feature_pickle, and test_target_pickle.

The evolve-feedforward.py is used to find the best regularization rate for each method. myUtile.py and visualize.py contains some useful method using in training. The result will be store in the result_pickle which contains the regularization rate, and accuracy for each iteration.

The evaluate.ipynb is used to evaluate the accuracy of NEAT, it read results from the result_pickel.

After select the best regularization rate, run.py will repeat the experiment for 30 times and the result will be stored in the result_pickle.

eve.ipynb is used to do the statistical analysis for the result and do the feature selection depend on the result.


