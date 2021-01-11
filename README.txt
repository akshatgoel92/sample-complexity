
# Overview

This folder contains code to reproduce each table in the submitted coursework report. We cover the structure of the folder first. Then we give Python commands to reproduce the tables in the report. 

# Structure 

src
 |
 |    
 +-- multiclass
 |   |  
 |   +-- kernel_perceptron.py
 |   +-- one_vs_one.py
 |   +-- logistic_regression.py
 |   +-- samme.py
 |   +-- knn.py
 |   +-- run.py
 |   +-- run_alternate.py
 |   +-- helpers.py  
 +-- complexity
 |   |  
 |   +-- vanilla_perceptron.py
 |   +-- one_nn.py
 |   +-- winnow.py
 |   +-- linear_regression.py
 |   +-- run_sample_complexity.py
 |   +-- helpers.py
 |
 |
 +-- misc
 |   +-- get_misc_results.py
 |   +-- knn_visualization.py
 |   +-- sammon.py
     +-- helpers.py

# Reproducing tables 

In all the commands below replace 'table_x' with 'table_1' for Table 1, 'table_2' for Table 2, and so on. There is no check to see whether the table no. is correct so please check the report to ensure this. Please note that the results for multiple runs are generated together. Please put in the table no. of the training set results table from the report to get both sets of results.

For part 1: Multi-Class Classification: 

	For one vs all results: python multiclass/run.py 'table_x'
	
	For one vs one results: python multiclass/one_vs_one.py 'table_x'
	
	For k-nearest neighbours results: python multiclass/run_alternate.py 'table_x'
	
	For SAMME results: python multiclass/run_alternate.py 'table_x'

For part 1: Multi-class Classification: 
	[After running the above]

	For confusion matrix and mistake analysis: python misc/get_final_results.py 
	
	For k-nearest neighbours data visualisations: python misc/knn_visualization.py

For part 2: Sample Complexity Estimation:
	
	For least squares: python complexity/run_sample_complexity.py 'linear_regression' 1 1 1
	
	For linear perceptron: python complexity/run_sample_complexity.py 'linear_perceptron' 1 1 1
	
	For winnow: python complexity/run_sample_complexity.py 'winnow' 1 1 1
	
	For 1-nn: python complexity/run_sample_complexity.py 'one_nn' 1 1 1 
    	
		 