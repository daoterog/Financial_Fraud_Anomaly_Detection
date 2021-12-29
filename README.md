# Anomaly Detection applied to Money Laundering Detection using Ensemble Learning

In this compressed folder you can find the resources used to work in the project. Every function developed 
as every step within the pipelines is well documented. There are plenty of comments that make understanding
the thought process used for the functions creation very intuitive.

# Contents:

- data folder: Contains five datasets generated using the Synthetizor tool developed 
by FinCrime Dynamics, a british company dedicated to study financial crime. The datasets
were generated using different sets of parameters (varying the number of individuals, businesses,
and fradulent transactions), however, they resemble in the way that the same patterns are seen in
every dataset.
- bagging_modelling.ipynb: A pipeline designed to experiment with a bagging approach developed by the
team. Different feature engineering, feature selection, and models were tested across multiple runs. The
notebook has integrated a storing function that allows to store the results of the experiments.
- exploratory_data_analysis.ipynb: In here the data is explored, several plots regarding the data relations
and distributions are made. Moreover, clustering using MiniBatchKMeans and PCA projections of the data 
was attempted as well. The later were used for several combinations of feature engineering and feature selection
techniques.
- stacking_representation_learning_modelling.ipynb: A pipeline designed to experiment using different feature engineering
techniques over different acquired frameworks. The pipelines allows to supervised and unsupervised models for experimentation.
The pipeline was developed to determine which framework performs the best, to test feature engineering techniques and model hyperparamters simultaneosly across frameworks. The main function contained within the pipeline is 'cross_validation_framework'.
The function allows to set if the passed model is supervised or unsupervised, to specify if the "stacking" (use several base 
learners to make predictions) or the regular (single model) framework is going to be used, to set the number of extra layers the 
ensemble is going to have (only applicable for stacking related methodologies), and to specify whether the results are going
to be stored or not. 
- helper_functions.py: here all the functions developed in order to build the explained frameworks are contained. The file was created 
in order to make the experimentation easier and maintain the notebooks clean.k
