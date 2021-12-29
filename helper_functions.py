import os
import shutil
import errno

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import itertools

from prince import PCA

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'font.size': 13})

from sklearn.model_selection import (KFold, StratifiedKFold, train_test_split)

from sklearn.metrics import (precision_score, recall_score, roc_auc_score, f1_score, roc_curve, 
                            auc, confusion_matrix, precision_recall_curve, PrecisionRecallDisplay, average_precision_score)
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier

from scikit_feature_master.skfeature.function.similarity_based.SPEC import spec, feature_ranking

def variables_amount_deviation(df, variable_list, column_names):
    
    """
    Creates variables that states the deviation from the median scaled by the MAD
    of a chosen variable. x1 = (x0 - median(x0))/mad(x0)
    
    Args:
        df: DataFrame.
        variable_list: list of variable for which the deviation is calculated.
        column_names: list of names of the new variables ['sender_name', 'receiver_name']
        
    Return:
        df: with appender columns
    """
    
    # Define Variables that are going to be used
    global_variables = variable_list + ['amount']
    sender_variables = ['sender_type', 'account_from'] + global_variables
    receiver_variables = ['receiver_type', 'account_to'] + global_variables
    
    # Remove duplicates
    sender_variables = list(dict.fromkeys(sender_variables))
    receiver_variables = list(dict.fromkeys(receiver_variables))
    
    # Get median of the value and group by the desired variables
    # The value is calculated for the sender, receiver, and S2R type.
    aux_sender = df[sender_variables].groupby(sender_variables[:-1]).median().reset_index()
    aux_receiver = df[receiver_variables].groupby(receiver_variables[:-1]).median().reset_index()
    
    # Create a dictionary that maps the median of the value spended by month
    # by each of the identified groups
    aux_dict_sender = aux_sender.groupby(['sender_type'] + variable_list).median().to_dict()['amount']
    aux_dict_receiver = aux_receiver.groupby(['receiver_type'] + variable_list).median().to_dict()['amount']
    
    # Create Series objects that store the mapped dictionary values
    sender_median_ammount = df[['sender_type'] + variable_list].apply(lambda x: aux_dict_sender[tuple(x)], axis=1)
    receiver_median_ammount = df[['receiver_type'] + variable_list].apply(lambda x: aux_dict_receiver[tuple(x)], axis=1)
    
    #Substract the transaction amount to state the deviation
    sender_median_deviation_ammount = sender_median_ammount.sub(df.amount)
    receiver_median_deviation_ammount = receiver_median_ammount.sub(df.amount)
    
    # Auxiliar median deviation
    df['sender_abs_median_deviation'] = sender_median_deviation_ammount.abs()
    df['receiver_abs_median_deviation'] = receiver_median_deviation_ammount.abs()
    
    # Get the median of the absolute median deviation for each group
    aux_sender = df[sender_variables[:-1] + ['sender_abs_median_deviation']]\
                        .groupby(sender_variables[:-1]).median().reset_index()
    aux_receiver = df[receiver_variables[:-1] + ['receiver_abs_median_deviation']]\
                        .groupby(receiver_variables[:-1]).median().reset_index()
    
    # Create dictionary to map values the median absolute deviation of each group
    aux_dict_sender = aux_sender.groupby(['sender_type'] + variable_list).median()\
                            .to_dict()['sender_abs_median_deviation']
    aux_dict_receiver = aux_receiver.groupby(['receiver_type'] + variable_list).median()\
                            .to_dict()['receiver_abs_median_deviation']
    
    # Dfine Series objects that store mad values
    sender_mad_ammount = df[['sender_type'] + variable_list].apply(lambda x: aux_dict_sender[tuple(x)], axis=1)
    receiver_mad_ammount = df[['receiver_type'] + variable_list].apply(lambda x: aux_dict_receiver[tuple(x)], axis=1)
    
    # Add Columns to dataframe and drop the auxiliar ones created
    df[column_names[0]] = sender_median_deviation_ammount/sender_mad_ammount
    df[column_names[1]] = receiver_median_deviation_ammount/receiver_mad_ammount
    
    df.drop(columns=['receiver_abs_median_deviation', 'sender_abs_median_deviation'], inplace=True)
    
    return df

def perform_feauture_selection(X, y=None, mode='spec', n=None, n_sample=10000):

    """
    Performs feature selection according to the specified method (mode). spec uses
    Spectral Graph Theory to perform unsupervised feature selection. (recomended)

    Args:
        X (Dataframe): dataset.
        y (Series): dependant variable (binary encoded, fraud=1).
        mode (String): method to use.
        n (int): number of features to keep.
        n_sample (int): number of observations to include in 'sample-spec'.

    returns:
        df_feature_selection (Dataframe): dataset with columns filtered.
    """
    
    if mode == 'sample-spec':
        
        # Get sample
        X = X.sample(n=n_sample)
        
        # Extract the ranks of the features
        ranks = feature_ranking(spec(X.to_numpy()))

        # Get number of columns to get
        n_features = int(np.ceil(len(X.columns)*0.6))

        # Get columns indices, select them, and store them in a new df
        features = ranks[:n_features]
        keep_columns = X.columns[features]
        df_feature_selection = X[keep_columns]
                
    elif mode == 'spec':
        
        # Define K-Fold Split to make the processing quicker
        kf = KFold(n_splits=int(X.shape[0]/10000), shuffle=True)

        # List to store ranks assigned using SPEC methodology
        iteration_ranks = []

        # Loop to get ranks of each fold
        for _, indices in kf.split(X):

            X_spec = X.to_numpy()[indices, :]
            ranks = feature_ranking(spec(X_spec))
            iteration_ranks.append(ranks)

        # List to append mode of the previous iterations
        features = []
        feature_matrix = np.matrix(iteration_ranks)

        if n == None or n > len(X.columns):
            n = int(np.ceil(0.6*len(X.columns)))
        
        for i in range(n):
            
            mode = pd.Series(feature_matrix[:,i].tolist()).value_counts().sort_values(ascending=False)
            
            for value, _ in mode.iteritems():
                
                try:
                    _ = features.index(value[0])
                except ValueError:
                    features.append(value[0])
                    break
                else:
                    continue
           

        # Get columns to keep
        keep_columns = X.columns[features]
        df_feature_selection = X[keep_columns]
    
    elif mode == 'kbest':

        selector = SelectKBest()
        df_feature_selection = pd.DataFrame(selector.fit_transform(X, y))

    else: 
        
        model = ExtraTreesClassifier()
        model.fit(X,y)
        if n == None or n > len(X.columns):
            n = int(np.ceil(0.6*len(X.columns)))

        columns_to_keep = pd.Series(model.feature_importances_, index=X.columns).nlargest(n).index.tolist()

        df_feature_selection = X[columns_to_keep]

    return df_feature_selection

def unsupervised_representation_learning(model_dict, X_train, X_test, threshold):
    
    """
    Takes a dictionary of representation learners and produces an anomaly score with each of the models.
    These scores are transalated to predictions using a specified threshold. The scores and the predictions
    are meant to be used to perform stacking (in this case also regarded as representation learning).

    Args:
        model_dict (Dictionary): representation learners dicionary.
        X_train (Dataframe): training dataset.
        X_test (Dataframe): testing dataset.
        threshold (float): threshold that states where to regard an observation as an anomaly (in [0,1]).

    Returns:
        df_decision_scores (Dataframe): Contains decision scores assigned by each learner (columns).
        df_decision_function (Dataframe): Contains decision function assigned by each learner (columns).
        df_training_prediction (Dataframe): Contains predictions assigned by each learner (columns). 
        df_testing_prediction (Dataframe): Contains predictions assigned by each learner (columns).
    """
    
    # Create DataFrame to store the decision scores, the decision function,
    # the training prediction, and the testing prediction
    df_decision_scores = pd.DataFrame(index=X_train.index)
    df_decision_function = pd.DataFrame(index=X_test.index)
    df_training_prediction = pd.DataFrame(index=X_train.index)
    df_testing_prediction = pd.DataFrame(index=X_test.index)
    
    # Initialize Scaler Object to change decision scores and decision function
    # values range to [0, 1] in order for it to be interpreted as a probability
    # and use it as a y_score for ROC curve
    minmax_scaler = MinMaxScaler()
    robust_scaler = RobustScaler()
    
    for model_key in model_dict.keys():
        
        # Get the model, fit it, get the decision scores for the trainig
        # dataset and get the decision function value for the test dataset
        # and store them in their corresponding DataFrame
        model = model_dict[model_key]
        model.fit(X_train)
        df_decision_scores[model_key] = minmax_scaler.fit_transform(robust_scaler.fit_transform(model.decision_scores_.reshape(-1,1)))
        df_decision_function[model_key] = minmax_scaler.fit_transform(robust_scaler.fit_transform(model.decision_function(X_test).reshape(-1,1)))
        
        # Create binary markers that indicate the models predictions for the 
        # training and testing dataset and aassign values
        training_outlier_marker = (df_decision_scores[model_key] >
                                  df_decision_scores[model_key].quantile(threshold))
        testing_outlier_marker = (df_decision_function[model_key] >
                                 df_decision_function[model_key].quantile(threshold))
        training_prediction = pd.Series([0]*X_train.shape[0], index=X_train.index)
        testing_prediction = pd.Series([0]*X_test.shape[0], index=X_test.index)
        training_prediction[training_outlier_marker] = 1
        testing_prediction[testing_outlier_marker] = 1
        df_training_prediction[model_key] = training_prediction
        df_testing_prediction[model_key] = testing_prediction

    # Replace possible nan or inf values in predictions
    df_decision_scores = df_decision_scores.replace([np.nan, np.inf, -np.inf], 0)
        
    return (df_decision_scores, df_decision_function, 
            df_training_prediction, df_testing_prediction)

def accuracy_tos(df_decision_scores, df_decision_function, 
                 df_training_prediction, df_testing_prediction, y_train, y_test, accuracy):
    
    """
    Selects the 60% of the decision scores and decision functions produced by the representation
    learner in order to keep the most representatives (more accurate and less correlated with the others).

    Args:
        df_decision_scores (Dataframe): Contains decision scores assigned by each learner (columns).
        df_decision_function (Dataframe): Contains decision function assigned by each learner (columns).
        df_training_prediction (Dataframe): Contains predictions assigned by each learner (columns). 
        df_testing_prediction (Dataframe): Contains predictions assigned by each learner (columns).
        y_train (Series): Real labels in training set.
        y_test (Series): Real labels in testing set.
        accuracy (String): accuracy metric to evaluate learners.

    Returns:
        list of features names that shhould be kept.
    
    """
    
    # List of scores
    score_list = []
    
    # Loop that iterates over each column
    for representation_learner in df_decision_scores.columns:
        
        # Get learner predictions and probs
        training_preds = df_training_prediction[representation_learner]
        testing_preds = df_testing_prediction[representation_learner]
        training_probs = df_decision_scores[representation_learner]
        testing_probs = df_decision_function[representation_learner]
        dec_scores = df_decision_scores[representation_learner]
        dec_function = df_decision_function[representation_learner]
        
        # If statement that indicaes what metric to use as feature selection
        # criteria 
        if accuracy == 'precision':
            acc_training = precision_score(y_train, training_preds)
            acc_testing = precision_score(y_test, testing_preds)
        elif accuracy == 'recall':
            acc_training = recall_score(y_train, training_preds)
            acc_testing = recall_score(y_test, testing_preds)
        elif accuracy == 'f1':
            acc_training = f1_score(y_train, training_preds)
            acc_testing = f1_score(y_test, testing_preds)
        else:
            precision_train, recall_train, _ = precision_recall_curve(y_train, dec_scores)
            precision_test, recall_test, _ = precision_recall_curve(y_test, dec_function)

            acc_training = auc(recall_train, precision_train)
            acc_testing = auc(recall_test, precision_test)
            
        # Calculate correlation metric between different predictions to rank
        # their similarity and sum them up
        training_correlation = 0
        testing_correlation = 0
        for other_representation_learner in df_decision_scores.columns:
            
            # Discard same learner
            if representation_learner == other_representation_learner:
                continue
            else:
                
                # Get the other learners probs
                pairwise_training_probs = df_decision_scores[other_representation_learner]
                pairwise_testing_probs = df_decision_function[other_representation_learner]
                
                # Get spearman correlation 
                pairwise_training_correlation, _ = spearmanr(pairwise_training_probs, training_probs)
                pairwise_testing_correlation, _ = spearmanr(pairwise_testing_probs, testing_probs)
                
                # Sum them
                training_correlation += abs(pairwise_training_correlation)
                testing_correlation += abs(pairwise_testing_correlation)
                
        # Calculate the value for each set
        learner_training_score = acc_training/training_correlation
        learner_testing_score = acc_testing/testing_correlation
        
        # Weight the scores accoring to the size of the sets (modifiable accoridng to problem)
        train_size = y_train.shape[0]
        test_size = y_test.shape[0]
        total_score = (train_size*learner_training_score + test_size*learner_testing_score)/(test_size + train_size)
        
        # Append score to list
        score_list.append(total_score)
        
    # Get 60% of top scores
    score_series = pd.Series(score_list, index=df_decision_scores.columns)
    n_features = int(np.ceil(0.6*len(score_list)))
    
    return score_series.nlargest(n_features).index.tolist()

def evaluate_supervised_metalearner(model, X_dict, y_train, y_test):

    """
    Evaluates supervised learner and calculates the F1 score, the precision, recall, and the auc.

    Args:
        model (Object): supervised classifier.
        X_dict (Dictionary): dictionary with datasets.
        y_train (Series): for training.
        y_test (Series): for testing.

    Returns:
        df_y_prob (Dataframe): Dataframe containing the predicted probabilities for each dataset.
        df_y_pred (Dataframe): Dataframe containing the predictions for each dataset.
        df_f1 (Dataframe): Dataframe containing the f1 score for each dataset.
        df_precision (Dataframe): Dataframe containing the precision for each dataset.
        df_recall (Dataframe): Dataframe containing the recall for each dataset.
        df_auc (Dataframe): Dataframe containing the auc for each dataset.
    """

    # Metrics dictionaries
    f1_dict_testing = {}
    precision_dict_testing = {}
    recall_dict_testing = {}
    auc_dict_testing = {}
    y_prob_dict = {}
    y_pred_dict = {}

    # Loop to iterate over datasets
    for dataset_key in X_dict:

        # Get dataset
        (X_train, X_test) = X_dict[dataset_key]

        # Fit the model, make predictions, and predict probabilities
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = pd.DataFrame(model.predict_proba(X_test))[1]

        # Get Metrics
        f1_dict_testing[dataset_key] = f1_score(y_test, y_pred)
        precision_dict_testing[dataset_key] = precision_score(y_test, y_pred)
        recall_dict_testing[dataset_key] = recall_score(y_test, y_pred)
        precision_test, recall_test, _ = precision_recall_curve(y_test, y_prob)
        auc_dict_testing[dataset_key] = auc(recall_test, precision_test)
        y_prob_dict[dataset_key] = y_prob
        y_pred_dict[dataset_key] = y_pred

    # DataFrames to store Metrics
    df_y_prob = pd.DataFrame(y_prob_dict, index=y_test.index)
    df_y_pred = pd.DataFrame(y_pred_dict, index=y_test.index)
    df_f1 = pd.DataFrame(f1_dict_testing, index=['f1'])
    df_precision = pd.DataFrame(precision_dict_testing, index=['precision'])
    df_recall = pd.DataFrame(recall_dict_testing, index=['recall'])
    df_auc = pd.DataFrame(auc_dict_testing, index=['auc'])

    return df_y_prob, df_y_pred, df_f1, df_precision, df_recall, df_auc 


def evaluate_metalearner(model, X_dict, y_train, y_test, threshold):
    
    """
    Evaluates metalearner performance within training and testing sets. Stores F1, Precision, Recall, AUC,
    training and testing predictions, and raw decision scores and decision functions assigned to each observation.

    Args:
        model (Object): anomaly detector or classifier.
        X_dict (Dictionary): dictionary with every dataset to test.
        y_train (Series): Real labels in training set.
        y_test (Series): Real labels in testing set.
        threshold (float): threshold that states where to regard an observation as an anomaly (in [0,1]).
        counter (int): to assign proper indec to Dataframe instances.
        mean_tpr_train (Dictionary): auxiliary dictionary to collect the mean tpr for trainig sets (used to plot mean ROC Curve).
        mean_tpr_test (Dictionery): auxiliary dictionary to collect the mean tpr for trainig sets (used to plot mean ROC Curve).

    Returns:
        f1_tuple: tuple of Dataframes containing f1 scores for each dataset form -> (training_set_df, testing_set_df).
        precision_tuple: tuple of Dataframes containing precision scores for each dataset form -> (training_set_df, testing_set_df).
        recall_tuple: tuple of Dataframes containing recall scores for each dataset form -> (training_set_df, testing_set_df).
        auc_tuple: tuple of Dataframes containing auc scores for each dataset form -> (training_set_df, testing_set_df).
        roc_tuple_train: tuple of Dataframes containing necesary information to plot roc curves for each dataset.
        roc_tuple_test: tuple of Dataframes containing necesary information to plot roc curves for each dataset.
        pred_tuple: tuple of Dataframes containing predictions for each dataset form -> (training_set_df, testing_set_df).
        raw_decision_tuple: tuple of Dataframes containing decision scores and functions for each dataset form -> (decision_scores_df, decision_function_df).
    
    """
    
    # Metrics dictionaries
    f1_dict_training = {}; f1_dict_testing = {}
    precision_dict_training = {}; precision_dict_testing = {}
    recall_dict_training = {}; recall_dict_testing = {}
    auc_dict_training = {}; auc_dict_testing = {}
    # fpr_train = {}; tpr_train = {}; fpr_test = {}; tpr_test = {}
    # tprs_train = {}; tprs_test = {}
    # mean_fpr = np.linspace(0, 1, 100)
    training_predictions_dict = {}; testing_predictions_dict = {}
    raw_decision_scores = pd.DataFrame(); raw_decision_function = pd.DataFrame()
    
    # Initialize Scaler object
    minmax_scaler = MinMaxScaler()
    robust_scaler = RobustScaler()
    
    for dataset_key in X_dict.keys():
        
        # Get dataset
        (X_train, X_test) = X_dict[dataset_key]
        
        # Fit model, get decision scores, and get decision function values
        model.fit(X_train)
        dec_scores = model.decision_scores_
        dec_function = model.decision_function(X_test)
        raw_decision_scores[dataset_key] = dec_scores
        raw_decision_function[dataset_key] = dec_function
        decision_scores = pd.DataFrame(minmax_scaler.fit_transform(robust_scaler.fit_transform(dec_scores.reshape(-1,1))), 
                                       index=X_train.index)
        decision_function = pd.DataFrame(minmax_scaler.fit_transform(robust_scaler.fit_transform(dec_function.reshape(-1,1))), 
                                         index=X_test.index)
        
        # Create binary markers that indicate the models predictions for the 
        # training and testing dataset and aassign values
        training_outlier_marker = (decision_scores[0] > decision_scores[0].quantile(threshold))
        testing_outlier_marker = (decision_function[0] > decision_function[0].quantile(threshold))
        training_prediction = pd.Series([0]*X_train.shape[0], index=X_train.index)
        testing_prediction = pd.Series([0]*X_test.shape[0], index=X_test.index)
        training_prediction[training_outlier_marker] = 1
        testing_prediction[testing_outlier_marker] = 1
        training_predictions_dict[dataset_key] = training_prediction
        testing_predictions_dict[dataset_key] = testing_prediction
        
        # Get metrics
        f1_dict_training[dataset_key] = f1_score(y_train, training_prediction) 
        f1_dict_testing[dataset_key] = f1_score(y_test, testing_prediction)
        precision_dict_training[dataset_key] = precision_score(y_train, training_prediction) 
        precision_dict_testing[dataset_key] = precision_score(y_test, testing_prediction)
        recall_dict_training[dataset_key] = recall_score(y_train, training_prediction) 
        recall_dict_testing[dataset_key] = recall_score(y_test, testing_prediction)

        precision_train, recall_train, _ = precision_recall_curve(y_train, dec_scores)
        precision_test, recall_test, _ = precision_recall_curve(y_test, dec_function)

        auc_dict_training[dataset_key] = auc(recall_train, precision_train) 
        auc_dict_testing[dataset_key] = auc(recall_test, precision_test)
        
        # Roc curve metrics Not going to be used
        # fpr_tr, tpr_tr, _ = roc_curve(y_train, decision_scores)
        # fpr_train[dataset_key], tpr_train[dataset_key] = [fpr_tr], [tpr_tr]
        # aux_train = np.interp(mean_fpr, fpr_tr, tpr_tr)
        # mean_tpr_train[dataset_key] += aux_train
        # mean_tpr_train[dataset_key][0] = 0.0
        # tprs_train[dataset_key] = [aux_train]
        
        # fpr_te, tpr_te, _ = roc_curve(y_test, decision_function)
        # fpr_test[dataset_key], tpr_test[dataset_key] = [fpr_te], [tpr_te]
        # aux_test = np.interp(mean_fpr, fpr_te, tpr_te)
        # mean_tpr_test[dataset_key] += aux_test
        # mean_tpr_test[dataset_key][0] = 0.0
        # tprs_test[dataset_key] = [aux_test]
    
    # DataFrames to store Metrics
    raw_decision_scores.index = X_train.index
    raw_decision_function.index = X_test.index
    raw_decision_tuple = (raw_decision_scores, raw_decision_function)
    f1_tuple = (pd.DataFrame(f1_dict_training, index=['f1']), 
                pd.DataFrame(f1_dict_testing, index=['f1']))
    precision_tuple = (pd.DataFrame(precision_dict_training, index=['precision']), 
                       pd.DataFrame(precision_dict_testing, index=['precision']))
    recall_tuple = (pd.DataFrame(recall_dict_training, index=['recall']), 
                    pd.DataFrame(recall_dict_testing, index=['recall']))
    auc_tuple = (pd.DataFrame(auc_dict_training, index=['auc']), 
                 pd.DataFrame(auc_dict_testing, index=['auc']))
    # roc_tuple_train = (pd.DataFrame(fpr_train, index=[counter]), 
    #              pd.DataFrame(tpr_train, index=[counter]),
    #                    mean_tpr_train,
    #              pd.DataFrame(tprs_train, index=[counter]))
    # roc_tuple_test = (pd.DataFrame(fpr_test, index=[counter]), 
    #              pd.DataFrame(tpr_test, index=[counter]),
    #                    mean_tpr_test,
    #              pd.DataFrame(tprs_test, index=[counter]))
    pred_tuple = (pd.DataFrame(training_predictions_dict, index=X_train.index),
                  pd.DataFrame(testing_predictions_dict, index=X_test.index))
        
    return (f1_tuple, precision_tuple, recall_tuple, auc_tuple, 
            pred_tuple, raw_decision_tuple)

def make_confusion_matrix(y_true, y_pred, fig, ax, precision, title, classes=None, 
                           norm=False, text_size=10): 

    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels.
    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.
    Args:
        y_true: Array of truth labels (must be same shape as y_pred).
        y_pred: Array of predicted labels (must be same shape as y_true).
        fig: figure object for plotting.
        ax: axis object for plotting.
        accuracy: accuracy score calculated in across_class_results.
        classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
        figsize: Size of output figure (default=(10, 10)).
        text_size: Size of output figure text (default=15).
        norm: normalize values or not (default=False).
        savefig: save confusion matrix to file (default=False).
    
    Returns:
        A labelled confusion matrix plot comparing y_true and y_pred.
    Example usage:
        make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10)
    """  
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax, ax=ax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    
    # Label the axes
    ax.set(title=f"{title} Confusion Matrix",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(n_classes), # create enough axis slots for each class
            yticks=np.arange(n_classes), 
            xticklabels=labels, # axes will labeled with class names (if they exist) or ints
            yticklabels=labels)
    
    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            ax.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)
        else:
            ax.text(j, i, f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)

def plotrocauc(auc_list, fpr, tpr, tprs, mean_auc, mean_fpr,
               mean_tpr, title, fig, ax):
    """
    Plot the ROC curve.
    Args:
        auc_list: list of auc values.
        fpr: false positive rate list.
        tpr: true positive rate list.
        mean_auc: value.
        mean_fpr: mean fpr list.
        mean_tpr: mean tpr list.
        title: model_name.
        path: to load the data.
    """

    # Plot ROC AUC Curves
    for i in range(len(fpr)):
        ax.plot(fpr[i], tpr[i], lw = 3, alpha = 0.5,
                label='ROC fold %d (area = %0.2f)' % (i, auc_list[i]))

    # Plot diagonal
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance',
            alpha=.8)

    # Plot Mean ROC AUC
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_list)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % 
            (mean_auc, std_auc),
            lw=2, alpha=1)
    
    # Plot Confidence Interval
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', 
                    alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    # Settings
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title=f'{title} ROC Curve')
    ax.legend(loc="lower right")

def make_precision_recall_curves(y_true, y_prob, index_list, auc_value, mode, ax):

    """
    Plots precision-recall curves for each fold of the cross validation.

    Args:
        y_true (Series): true labels.
        y_prob (Series): probability of belonging to a class.
        index_list (list):  indexes of each cross validation fold.
        auc_value (float): mean auc value of the precision recall curve.
        mode (String): specify 
        ax: ax used to plot testing precision-recall curve.

    Output:
        Precision-Recall curves for training and testing sets.
    """

    for i in range(len(index_list)):

        # Get indexes of each split of cross validation
        indices = index_list[i]

        # Get predictions and true values of each cross validation split
        y_probs = y_prob.loc[indices].replace([np.nan, -np.inf, np.inf], 0).tolist()
        y_test = y_true.loc[indices].tolist()

        # Get Metrics
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        average_precision = average_precision_score(y_test, y_probs)
        acc = auc(recall, precision)

        # Plot 
        display = PrecisionRecallDisplay(precision=precision, recall=recall, 
                                                average_precision=average_precision)

        display.plot(ax=ax, name=f'Precision-Recall Fold {i+1}, AUC={np.round(acc, 2)}')

        ax.set_title(f'{mode} Precision-Recall Curve, Mean AUC={np.round(auc_value, 2)}')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc='best')

def make_supervised_cross_validation_plots(df_pred, df_true, df_prob, df_index, df_auc, 
                                            datasets, metalearner_name, path):

    """
    Makes supervised Cross-Validation plots.

    Args:
        df_pred (Dataframe): Dataframe of predicted values.
        df_true (Dataframe): Dataframe of true values.
        df_prob (Dataframe): Dataframe of probaabilities of being a fraud.
        df_index (Dataframe): Dataframe containing indexes of each cross validation fold.
        df_auc (Dataframe): Dataframe containing mean auc value for each dataset
        datsets (list): list of dataset keys.
        metalearner_name (String): model identifier for plotting.
    """

    # Loop to iterate over Datasets
    for dataset_key in datasets:

        y_true = df_true['fraud']
        y_pred = df_pred[dataset_key].tolist()
        y_prob = df_prob[dataset_key]
        auc = df_auc[dataset_key]

        # Intialize plot 
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot Confusion Matrix
        make_confusion_matrix(y_true.tolist(), y_pred, fig, ax[0], 0, 'Supervised',
                              classes=['transaction','fraud'], norm=True)

        # Precision Recall Curve
        make_precision_recall_curves(y_true, y_prob, df_index, auc, 'Supervised', ax[1])

        # Title and Saving
        title = dataset_key + '-' + metalearner_name
        fig.suptitle(title)
        
        # Save figure if specified
        if path != None:
            fig.savefig(os.path.join(path, title + '.png'), facecolor='w', dpi=100)
        
        plt.show()

def make_plots_cross_validation_plots(pred_tuple, true_tuple, decision_tuple, index_tuple, datasets, k, 
                                      metalearner_name, rep_learner_codes, auc_tuple, path):
    
    """
    Makes plots for cross validation.

    Args:
        roc_tuple_train: tuple of Dataframes containing necesary information to plot roc curves for each dataset.
        roc_tuple_test: tuple of Dataframes containing necesary information to plot roc curves for each dataset.
        pred_tuple: tuple of Dataframes containing predictions for each dataset form -> (training_set_df, testing_set_df).
        true_tuple: tuple of Dataframes containing true labels for each dataset form -> (training_set_df, testing_set_df).
        decision_tuple: tuple of decision scores and functions form -> (training_Series, testing_Series).
        index_tuple: tuple of indexes of each fold form -> (training_list, testing_list).
        datasets (list): list of indicators.
        k (int): number of splits
        metalearner_name (String): name of learner.
        rep_learner_codes (String): codes for representation learners.
        path (String): path to store plots.

    Outputs:
        Confusion Matrix, ROC Curve, and Precision-Recall Curve
    """
    
    # Unpack parameters from tuples
    # (fpr_df_train, tpr_df_train, mean_tpr_train, 
    #          tprs_df_train, auc_df_training) = roc_train_tuple
    # (fpr_df_test, tpr_df_test, mean_tpr_test, 
    #          tprs_df_test, auc_df_testing) = roc_test_tuple
    (preds_df_train, preds_df_test) = pred_tuple
    (true_df_train, true_df_test) = true_tuple
    (df_raw_decision_scores, df_raw_decision_function)= decision_tuple 
    (auc_training, auc_testing) = auc_tuple
    
    # Loop to iterate over Datasets
    for dataset_key in datasets:
        
        # Unpack parameters from Dataframes
        # auc_list_train = auc_df_training[dataset_key].tolist()
        # fpr_train = fpr_df_train[dataset_key].tolist()
        # tpr_train = tpr_df_train[dataset_key].tolist()
        # tprs_train = tprs_df_train[dataset_key].tolist()
        # mean_fpr = np.linspace(0, 1, 100)
        # mean_tpr_train_val = mean_tpr_train[dataset_key]
        # mean_tpr_train_val /= k
        # mean_tpr_train_val[-1] = 1.0
        # mean_auc_train = auc(mean_fpr, mean_tpr_train_val)
        # auc_list_test = auc_df_testing[dataset_key].tolist()
        # fpr_test = fpr_df_test[dataset_key].tolist()
        # tpr_test = tpr_df_test[dataset_key].tolist()
        # tprs_test = tprs_df_test[dataset_key].tolist()
        # mean_fpr = np.linspace(0, 1, 100)
        # mean_tpr_test_val = mean_tpr_test[dataset_key]
        # mean_tpr_test_val /= k
        # mean_tpr_test_val[-1] = 1.0
        # mean_auc_test = auc(mean_fpr, mean_tpr_test_val)
        y_true_train = true_df_train['fraud']
        y_pred_train = preds_df_train[dataset_key].tolist()
        y_true_test = true_df_test['fraud']
        y_pred_test = preds_df_test[dataset_key].tolist()
        raw_decision_scores = df_raw_decision_scores[dataset_key]
        raw_decision_function = df_raw_decision_function[dataset_key]
        new_auc_tuple = (auc_training[dataset_key], auc_testing[dataset_key])
        
        # Intialize plot 
        fig, ax = plt.subplots(2, 2, figsize=(20, 20))
        
        # Plot Confusion Matrix
        make_confusion_matrix(y_true_train.tolist(), y_pred_train, fig, ax[0, 0], 0, 'Training',
                              classes=['transaction','fraud'], norm=True)
        make_confusion_matrix(y_true_test.tolist(), y_pred_test, fig, ax[1, 0], 0, 'Testing',
                              classes=['transaction','fraud'], norm=True)
        
        # ROC Curve # Not taken into account
        # plotrocauc(auc_list_train, fpr_train, tpr_train, tprs_train, mean_auc_train, 
        #            mean_fpr, mean_tpr_train_val, 'Training', fig, ax[0, 1])
        # plotrocauc(auc_list_test, fpr_test, tpr_test, tprs_test, mean_auc_test, 
        #            mean_fpr, mean_tpr_test_val, 'Testing', fig, ax[1, 1])

        # Create new parameters to plot precision recall curve

        # Precision Recall Curve
        make_precision_recall_curves(y_true_train, raw_decision_scores, index_tuple[0], auc_training[dataset_key], 'Training', ax[0, 1])
        make_precision_recall_curves(y_true_test, raw_decision_function, index_tuple[1], auc_testing[dataset_key], 'Testing', ax[1, 1])
        
        # Title and Saving
        title = dataset_key + '-' + rep_learner_codes + '-' + metalearner_name
        fig.suptitle(title)
        
        # Save figure if specified
        if path != None:
            fig.savefig(os.path.join(path, title + '.png'), facecolor='w', dpi=100)
        
        plt.show()
        
def cross_validation_framework(metalearner, X, y, k, metalearner_name='', threshold=0.99, accuracy='precision',  
                                representation_learning_model_dict=None, rep_learner_codes='', mode='stacking', 
                                supervised=False, n_layers=0, path=None):
    
    """
    Performs cross validation according to the specified framework (mode).

    Args:
        model (Object): anomaly detector or classifier.
        X (Dataframe): dataset (independet variables).
        y (Series): independent variables.
        k (int): number of folds for cross validatioon.
        metalearner_name (String): name of learner.
        threshold (float): threshold that states where to regard an observation as an anomaly (in [0,1]).
        accuracy (String): accuracy metric to evaluate learners.
        representation_learning_model_dict (Dictionary): representation learners dicionary.
        rep_learner_codes (String): representation learner codes,
        mode (String): cross validation framework.
        n_layers (int): number of additional layers of stacking.
        path (String): path to save figures and results.

    Returns:
        training_results (Dataframe): DataFrame containing training results for each dataset. 
        testing_results (Dataframe): DataFrame containing testing results for each dataset.
         
    """
    
    # DataFrames to store Metrics
    f1_df_training = pd.DataFrame(); f1_df_testing = pd.DataFrame()
    precision_df_training = pd.DataFrame(); precision_df_testing = pd.DataFrame()
    recall_df_training = pd.DataFrame(); recall_df_testing = pd.DataFrame()
    auc_df_training = pd.DataFrame(); auc_df_testing = pd.DataFrame()
    # fpr_df_train = pd.DataFrame(); tpr_df_train = pd.DataFrame()
    # tprs_df_train = pd.DataFrame()
    # fpr_df_test = pd.DataFrame(); tpr_df_test = pd.DataFrame()
    # tprs_df_test = pd.DataFrame()
    preds_df_train = pd.DataFrame(); preds_df_test = pd.DataFrame()
    true_df_train = pd.DataFrame(); true_df_test = pd.DataFrame()
    train_indexes_list = []; test_indexes_list = []
    df_raw_decision_scores = pd.DataFrame(); df_raw_decision_function = pd.DataFrame()
    
    # Initialize Stratified Splitter object
    skf = StratifiedKFold(n_splits=k)
    
    # Auxiliar objects
    if n_layers == 0:
        dataset_keys = ['Regular', 'Decision-Scores', 'Filetered-Decision-Scores', 'Combined', 'Combined-Filtered']
    else:
        dataset_keys = ['Regular', 'Decision-Scores', 'Boosting-Decision-Scores', 'Combined-First', 'Combined-Boosted']
    # mean_tpr_train = dict(zip(dataset_keys,[0.0]*len(dataset_keys)))
    # mean_tpr_test = dict(zip(dataset_keys,[0.0]*len(dataset_keys)))
    
    # Loop for Cross Validation
    for train_indices, test_indices in skf.split(X, y):

        # Append indexes to list
        train_indexes_list.append(train_indices)
        test_indexes_list.append(test_indices)
        
        # Split the dataset
        X_train = X.loc[train_indices, :]
        X_test = X.loc[test_indices, :]
        y_train = y.loc[train_indices]
        y_test = y.loc[test_indices]

        if mode != 'stacking':

            # Build new X_dict
            new_X_dict = {'Regular': (X_train, X_test)}
            dataset_keys = ['Regular']
            
            # Evaluate Meta-learner over each dataset
            if not supervised:

                (f1_tuple, precision_tuple, recall_tuple, 
                        auc_tuple, pred_tuple, raw_decision_tuple) = evaluate_metalearner(metalearner, new_X_dict, y_train, 
                                                                                    y_test, threshold)
            
            else:
                (raw_decision_function, df_test_preds, df_f1_testing, 
                        df_precision_testing, df_recall_testing, df_auc_testing) = evaluate_supervised_metalearner(metalearner, new_X_dict, y_train, y_test)

        else: 

            # Create representation learning dataset
            (df_decision_scores, df_decision_function, 
                df_training_prediction, df_testing_prediction) = unsupervised_representation_learning(representation_learning_model_dict, 
                                                                                                    X_train, 
                                                                                                    X_test, 
                                                                                                    threshold)

            # Perform Feature Selection
            features_to_keep = accuracy_tos(df_decision_scores, df_decision_function, df_training_prediction, 
                                            df_testing_prediction, y_train, y_test, accuracy)
            
            # Filter Learners
            df_filtered_decision_scores = df_decision_scores[features_to_keep]
            df_filtered_decision_function = df_decision_function[features_to_keep]
            df_filtered_training_prediction = df_training_prediction[features_to_keep]
            df_filtered_testing_prediction = df_testing_prediction[features_to_keep]
            
            # Build each dataset
            X_comb_training = X_train.join(df_decision_scores, how='outer')
            X_comb_testing = X_test.join(df_decision_function, how='outer')

            # Enable additionaal layers of stacking
            for j in range(n_layers):

                print(f'{j+1}th layer of stacking')

                # Create representation learning dataset using the combined dataset (because is the second layer)
                (df_decision_scores_n, df_decision_function_n, 
                df_training_prediction_n, df_testing_prediction_n) = unsupervised_representation_learning(representation_learning_model_dict, 
                                                                                                    X_comb_training, 
                                                                                                    X_comb_testing, 
                                                                                                    threshold)

                # Rename columns to avoid overlapping
                df_decision_scores_n.columns = [learner + '-' + str(j) for learner in df_decision_scores_n.columns]
                df_decision_function_n.columns = [learner + '-' + str(j) for learner in df_decision_function_n.columns]
                df_training_prediction_n.columns = [learner + '-' + str(j) for learner in df_training_prediction_n.columns]
                df_testing_prediction_n.columns = [learner + '-' + str(j) for learner in df_testing_prediction_n.columns]

                # Join Decision Scores
                df_filtered_decision_scores = df_filtered_decision_scores.join(df_decision_scores_n, how='outer')
                df_filtered_decision_function = df_filtered_decision_function.join(df_decision_function_n, how='outer')
                df_filtered_training_prediction = df_filtered_training_prediction.join(df_training_prediction_n, how='outer')
                df_filtered_testing_prediction = df_filtered_testing_prediction.join(df_testing_prediction_n, how='outer')

                # Perform Feature Selection
                features_to_keep_n = accuracy_tos(df_filtered_decision_scores, df_filtered_decision_function, df_filtered_training_prediction, 
                                                df_filtered_testing_prediction, y_train, y_test, accuracy)

                # Filter Learners
                df_filtered_decision_scores = df_filtered_decision_scores[features_to_keep_n]
                df_filtered_decision_function = df_filtered_decision_function[features_to_keep_n]
                df_filtered_training_prediction = df_filtered_training_prediction[features_to_keep_n]
                df_filtered_testing_prediction = df_filtered_testing_prediction[features_to_keep_n]

                # Build each dataset
                X_comb_training = X_train.join(df_filtered_decision_scores, how='outer')
                X_comb_testing = X_test.join(df_filtered_decision_function, how='outer')

            if n_layers == 0:
                X_comb_training_filtered = X_train.join(df_filtered_decision_scores, how='outer')
                X_comb_testing_filtered = X_test.join(df_filtered_decision_function, how='outer')
            else:
                X_comb_training = X_train.join(df_decision_scores, how='outer')
                X_comb_testing = X_test.join(df_decision_function, how='outer')
                X_comb_training_filtered = X_train.join(df_filtered_decision_scores, how='outer')
                X_comb_testing_filtered = X_test.join(df_filtered_decision_function, how='outer')


            # Build New Dataset Dictionary
            dataset_values = [(X_train, X_test), (df_decision_scores, df_decision_function),
                            (df_filtered_decision_scores, df_filtered_decision_function),
                            (X_comb_training, X_comb_testing), (X_comb_training_filtered, X_comb_testing_filtered)]
            new_X_dict = dict(zip(dataset_keys, dataset_values))
            
            if not supervised:

                # Evaluate Meta-learner over each dataset
                (f1_tuple, precision_tuple, recall_tuple, 
                        auc_tuple, pred_tuple, raw_decision_tuple) = evaluate_metalearner(metalearner, new_X_dict, y_train, 
                                                                                    y_test, threshold)

            else:

                (raw_decision_function, df_test_preds, df_f1_testing, 
                        df_precision_testing, df_recall_testing, df_auc_testing) = evaluate_supervised_metalearner(metalearner, new_X_dict, y_train, y_test)
            
        # Extract and append datasets
        if not supervised:

            (df_f1_training, df_f1_testing) = f1_tuple
            (df_precision_training, df_precision_testing) = precision_tuple
            (df_recall_training, df_recall_testing) = recall_tuple
            (df_auc_training, df_auc_testing) = auc_tuple
            # (df_fpr_train, df_tpr_train, mean_tpr_train, df_tprs_train) = roc_tuple_train
            # (df_fpr_test, df_tpr_test, mean_tpr_test, df_tprs_test) = roc_tuple_test
            (df_train_preds, df_test_preds) = pred_tuple
            (raw_decision_scores, raw_decision_function)= raw_decision_tuple 
            f1_df_training = f1_df_training.append(df_f1_training)
            precision_df_training = precision_df_training.append(df_precision_training)
            recall_df_training = recall_df_training.append(df_recall_training)
            auc_df_training = auc_df_training.append(df_auc_training)
            # fpr_df_train = fpr_df_train.append(df_fpr_train)
            # tpr_df_train = tpr_df_train.append(df_tpr_train)
            # tprs_df_train = tprs_df_train.append(df_tprs_train)
            # fpr_df_test = fpr_df_test.append(df_fpr_test)
            # tpr_df_test = tpr_df_test.append(df_tpr_test)
            # tprs_df_test = tprs_df_test.append(df_tprs_test)
            # df_train_preds.index = train_indices
            # df_test_preds.index = test_indices
            preds_df_train = preds_df_train.append(df_train_preds)
            true_df_train = true_df_train.append(pd.DataFrame(y_train))
            df_raw_decision_scores = df_raw_decision_scores.append(raw_decision_scores)
            
        f1_df_testing = f1_df_testing.append(df_f1_testing)
        precision_df_testing = precision_df_testing.append(df_precision_testing)
        recall_df_testing = recall_df_testing.append(df_recall_testing)
        auc_df_testing = auc_df_testing.append(df_auc_testing)
        preds_df_test = preds_df_test.append(df_test_preds)
        true_df_test = true_df_test.append(pd.DataFrame(y_test))
        df_raw_decision_function = df_raw_decision_function.append(raw_decision_function)
    

    # Get mean of every metric
    f1_testing_mean = pd.DataFrame(f1_df_testing.mean()).transpose()
    precision_testing_mean = pd.DataFrame(precision_df_testing.mean()).transpose()
    recall_testing_mean = pd.DataFrame(recall_df_testing.mean()).transpose()
    auc_testing_mean = pd.DataFrame(auc_df_testing.mean()).transpose()

    # Non supervised case
    if not supervised:

        f1_training_mean = pd.DataFrame(f1_df_training.mean()).transpose()
        precision_training_mean = pd.DataFrame(precision_df_training.mean()).transpose()
        recall_training_mean = pd.DataFrame(recall_df_training.mean()).transpose()
        auc_training_mean = pd.DataFrame(auc_df_training.mean()).transpose()
        
    
    # Concatenate the dataframes and turn them into two dataframes
    idx = ['precision', 'recall', 'f1', 'auc']

    testing_results = pd.concat([precision_testing_mean, recall_testing_mean, 
                                  f1_testing_mean, auc_testing_mean], axis=0)
    testing_results.index = idx

    # Non supervised case
    if not supervised:
        training_results = pd.concat([precision_training_mean, recall_training_mean, 
                                    f1_training_mean, auc_training_mean], axis=0)
        training_results.index = idx

    # Save dataframes in case it is specified
    if path != None:

        testing_results.to_csv(os.path.join(path,'testing_results.csv'))

        if not supervised:
            training_results.to_csv(os.path.join(path,'training_results.csv'))

    # Parameters for Plots
    # roc_train_tuple = (fpr_df_train, tpr_df_train, mean_tpr_train, tprs_df_train, auc_df_training)
    # roc_test_tuple = (fpr_df_test, tpr_df_test, mean_tpr_test, tprs_df_test, auc_df_testing)
    if not supervised:

        pred_tuple = (preds_df_train, preds_df_test)
        true_tuple = (true_df_train, true_df_test)
        decision_tuple = (df_raw_decision_scores, df_raw_decision_function)
        index_tuple = (train_indexes_list, test_indexes_list)
        auc_tuple = (training_results.loc['auc', :], testing_results.loc['auc', :])

        # Make plots
        make_plots_cross_validation_plots(pred_tuple, true_tuple, decision_tuple, index_tuple, 
                                            dataset_keys, k, metalearner_name, rep_learner_codes, auc_tuple, path)

        return training_results, testing_results

    else:

        make_supervised_cross_validation_plots(preds_df_test, true_df_test, df_raw_decision_function, test_indexes_list, testing_results.loc['auc', :], 
                                            dataset_keys, metalearner_name, path)

        return pd.DataFrame() ,testing_results


def cross_validation_bagging(model_dict, X_dict, y, k, threshold, model_codes='', path=None):

    """
    Cross validation routine for bagging apprach. It plots the confusion matrix and the precision-recall
    curve for training and testing. It sumarizes the results for each dataset calculating the mean
    f1 score, precision, recall, and auc (of precision-recall) and stores them in a Dataframe.
    
    Args:
        model_dict (Dictionary): dictionary of models to make the prediction.
        X_dict (Dictionary): dictionary of datasets (Dataframes) to test.
        y (Series): dependant variable (binary encoded, outlier=1).
        k: number of splits for cross validation.
        threshold (int): threshold value to consider an outlier an anomaly.
        model_codes (String): Identifier for the used models.
        path (String): path to store results. 
        
    Returns:
        df_train_results (Dataframe): Dataframe sumarizing train scores for each of the tested datasets.
        df_test_results (Dataframe): Dataframe sumarizing test scores for each of the tested datasets.
    """

    # Desired Metrics 
    train_f1_dict = {}; train_precision_dict = {}
    train_recall_dict = {}; train_auc_dict = {}
    test_f1_dict = {}; test_precision_dict = {}
    test_recall_dict = {}; test_auc_dict = {}

    # Initialize splitter object
    skf = StratifiedKFold(n_splits=k)

    # Initialize Scaler object
    scaler = MinMaxScaler()

    # Loop to iterate over datasets
    for dataset_key in X_dict.keys():

        # Get the dataset
        X = X_dict[dataset_key]

        # Define lists to store values
        train_f1_list = []; train_precision_list = []
        train_recall_list = []; train_auc_list = []
        test_f1_list = []; test_precision_list = []
        test_recall_list = []; test_auc_list = []
        df_train_true = pd.DataFrame(); df_test_true = pd.DataFrame()
        train_pred_list = []; test_pred_list = []
        df_train_prob = pd.DataFrame(); df_test_prob = pd.DataFrame()
        train_index_list = []; test_index_list = []

        # Loop to iterate over every split
        for i, (train, test) in enumerate(skf.split(X, y)):

            # Separate in train and test samples
            X_train = X.iloc[train, :]
            X_test = X.iloc[test, :]
            y_train = y.iloc[train]
            y_test = y.iloc[test]
            train_index_list.append(train)
            test_index_list.append(test)
            df_train_true = df_train_true.append(pd.DataFrame(y_train, index=train))
            df_test_true = df_test_true.append(pd.DataFrame(y_test, index=test))

            # DataFrame to store predictions
            df_train_pred = pd.DataFrame()
            df_test_pred = pd.DataFrame()
            stand_decision_scores = pd.DataFrame()
            stand_decision_function = pd.DataFrame()

            # Iterate over all the Defined models
            for model_key in model_dict.keys():
                
                # Get model, fit it and make prediction
                model = model_dict[model_key]
                model.fit(X_train)
                raw_decision_scores = pd.Series(model.decision_scores_, index=train)
                raw_decision_function = pd.Series(model.decision_function(X_test), index=test)                
                train_pred_marker = raw_decision_scores > raw_decision_scores.quantile(threshold)
                test_pred_marker = raw_decision_function > raw_decision_function.quantile(threshold)
                train_pred = pd.Series([0]*X_train.shape[0], index=train)
                test_pred = pd.Series([0]*X_test.shape[0], index=test)
                train_pred[train_pred_marker] = 1
                test_pred[test_pred_marker] = 1

                # Append predictions
                df_train_pred[model_key] = train_pred
                df_test_pred[model_key] = test_pred
                stand_decision_scores[model_key] = pd.DataFrame(scaler.fit_transform(model.decision_scores_.reshape(-1,1)))
                stand_decision_function[model_key] = pd.DataFrame(scaler.fit_transform(model.decision_function(X_test).reshape(-1,1)))

            # Check make new predictions by majority of votes
            n_votes = len(model_dict.keys())/2
            final_train_pred = df_train_pred.sum(1).apply(lambda x: 1 if x > n_votes else 0)
            final_test_pred = df_test_pred.sum(1).apply(lambda x: 1 if x > n_votes else 0)
            final_decision_scores = stand_decision_scores.mean(1)
            final_decision_function = stand_decision_function.mean(1)
            train_pred_list.extend(final_train_pred.tolist())
            test_pred_list.extend(final_test_pred.tolist())
            df_train_prob = df_train_prob.append(pd.DataFrame(final_decision_scores, index=train))
            df_test_prob = df_test_prob.append(pd.DataFrame(final_decision_function, index=test))

            # Evaluate Predictions
            train_f1_list.append(f1_score(y_train, final_train_pred))
            test_f1_list.append(f1_score(y_test, final_test_pred))
            train_precision_list.append(precision_score(y_train, final_train_pred))
            test_precision_list.append(precision_score(y_test, final_test_pred))
            train_recall_list.append(recall_score(y_train, final_train_pred))
            test_recall_list.append(recall_score(y_test, final_test_pred))
            
            precision_train, recall_train, _ = precision_recall_curve(y_train, final_decision_scores)
            precision_test, recall_test, _ = precision_recall_curve(y_test, final_decision_function)

            train_auc_list.append(auc(recall_train, precision_train))
            test_auc_list.append(auc(recall_test, precision_test))

        # Store mean metrics in dictionaries
        train_f1_dict[dataset_key] = np.mean(train_f1_list)
        test_f1_dict[dataset_key] = np.mean(test_f1_list)
        train_precision_dict[dataset_key] = np.mean(train_precision_list)
        test_precision_dict[dataset_key] = np.mean(test_precision_list)
        train_recall_dict[dataset_key] = np.mean(train_recall_list)
        test_recall_dict[dataset_key] = np.mean(test_recall_list)
        train_auc_dict[dataset_key] = np.mean(train_auc_list)
        test_auc_dict[dataset_key] = np.mean(test_auc_list)

        # Make plots
        fig, ax = plt.subplots(2, 2, figsize=(20, 20))

        # Plot Confusion Matrix
        print(df_train_true.columns)
        make_confusion_matrix(df_train_true['fraud'].tolist(), train_pred_list, fig, ax[0, 0], 0, 'Training',
                              classes=['transaction','fraud'], norm=True)
        make_confusion_matrix(df_test_true['fraud'].tolist(), test_pred_list, fig, ax[1, 0], 0, 'Testing',
                              classes=['transaction','fraud'], norm=True)

        # Precision Recall Curve
        make_precision_recall_curves(df_train_true['fraud'], df_train_prob[0], train_index_list, train_auc_dict[dataset_key], 'Training', ax[0, 1])
        make_precision_recall_curves(df_train_true['fraud'], df_train_prob[0], test_index_list, test_auc_dict[dataset_key], 'Testing', ax[1, 1])
        
        # Title and Saving
        title = dataset_key + '-' + model_codes
        fig.suptitle(title)
        
        # Save figure if specified
        if path != None:
            fig.savefig(os.path.join(path, title + '.png'), facecolor='w', dpi=100)
        
        plt.show()
    
    # Summarize Results
    df_f1_train = pd.DataFrame(train_f1_dict, index=[0]); df_f1_test = pd.DataFrame(test_f1_dict, index=[0])
    df_precision_train = pd.DataFrame(train_precision_dict, index=[0]); df_precision_test = pd.DataFrame(test_precision_dict, index=[0])
    df_recall_train = pd.DataFrame(train_recall_dict, index=[0]); df_recall_test = pd.DataFrame(test_recall_dict, index=[0])
    df_auc_train = pd.DataFrame(train_auc_dict, index=[0]); df_auc_test = pd.DataFrame(test_auc_dict, index=[0])

    idx = ['precision', 'recall', 'f1', 'auc']
    df_train_results = pd.concat([df_f1_train, df_precision_train, 
                                    df_recall_train, df_auc_train], index=idx)
    df_test_results = pd.concat([df_f1_test, df_precision_test, 
                                    df_recall_test, df_auc_test], index=idx)

    if path:
        df_train_results.to_csv(os.path.join(path,'training_results.csv'))
        df_test_results.to_csv(os.path.join(path,'testing_results.csv'))

    return df_train_results, df_test_results

def make_prediction(metalearner, X_tuple, y_tuple, threshold, accuracy,  
                    representation_learning_model_dict, mode, n_layers):
    
    """
    Makes predictions using the specified methodology.
    
    Args:
        model (Object): anomaly detector or classifier.
        X_tuple: tuple containing (X_train, X_test).
        y_tuple: tuple containing (y_train, y_test).
        threshold (float): threshold that states where to regard an observation as an anomaly (in [0,1]).
        accuracy (String): accuracy metric to evaluate learners.
        representation_learning_model_dict (Dictionary): representation learners dicionary.
        mode (String): cross validation framework.
        n_layers (int): number of additional layers of stacking.
        
    Returns:
        pred_tuple: tuple of Dataframes containing predictions for each dataset form -> (training_set_df, testing_set_df).
        X_dict (Dictionary): dictionary with used_dictionary.
    
    """
    
    # Unpack tuple parameters
    (X_train, X_test) = X_tuple
    (y_train, y_test) = y_tuple
    
    # Define prediction tuple (statement that is going to be returned)
    pred_tuple = 0
    
    # Define auxiliar dictionary in order to use the evaluate_metalearner function
    X_dict = {}
    
    # Choose prediction mode
    if mode == 'regular':
        
        # Build dataset dictionary (necessary in cross_valiation_framework as well)
        X_dict = {mode: (X_train, X_test)}
        
        # Evaluate learner and make predictions
        (_, _, _, _, pred_tuple, raw_decision_tuple) = evaluate_metalearner(metalearner, X_dict, y_train, y_test, threshold)
        
    else:
        
        # Check if representation learner are defined
        if representation_learning_model_dict != None:
            
            # Create representation learning dataset
            (df_decision_scores, df_decision_function, 
                df_training_prediction, df_testing_prediction) = unsupervised_representation_learning(representation_learning_model_dict, 
                                                                                                    X_train, 
                                                                                                    X_test, 
                                                                                                    threshold)
            
            # Perform Feature Selection
            features_to_keep = accuracy_tos(df_decision_scores, df_decision_function, df_training_prediction, 
                                            df_testing_prediction, y_train, y_test, accuracy)
            
            # Filter Learners
            df_filtered_decision_scores = df_decision_scores[features_to_keep]
            df_filtered_decision_function = df_decision_function[features_to_keep]
            df_filtered_training_prediction = df_training_prediction[features_to_keep]
            df_filtered_testing_prediction = df_testing_prediction[features_to_keep]
    
            # Build each dataset
            X_comb_training = X_train.join(df_decision_scores, how='outer')
            X_comb_testing = X_test.join(df_decision_function, how='outer')
            
            # Enable additionaal layers of stacking
            for j in range(n_layers):

                print(f'{j+1}th layer of stacking')

                # Create representation learning dataset using the combined dataset (because is the second layer)
                (df_decision_scores_n, df_decision_function_n, 
                df_training_prediction_n, df_testing_prediction_n) = unsupervised_representation_learning(representation_learning_model_dict, 
                                                                                                    X_comb_training, 
                                                                                                    X_comb_testing, 
                                                                                                    threshold)
                
                # Rename columns to avoid overlapping
                df_decision_scores_n.columns = [learner + '-' + str(j) for learner in df_decision_scores_n.columns]
                df_decision_function_n.columns = [learner + '-' + str(j) for learner in df_decision_function_n.columns]
                df_training_prediction_n.columns = [learner + '-' + str(j) for learner in df_training_prediction_n.columns]
                df_testing_prediction_n.columns = [learner + '-' + str(j) for learner in df_testing_prediction_n.columns]

                # Join Decision Scores
                df_filtered_decision_scores = df_filtered_decision_scores.join(df_decision_scores_n, how='outer')
                df_filtered_decision_function = df_filtered_decision_function.join(df_decision_function_n, how='outer')
                df_filtered_training_prediction = df_filtered_training_prediction.join(df_training_prediction_n, how='outer')
                df_filtered_testing_prediction = df_filtered_testing_prediction.join(df_testing_prediction_n, how='outer')

                # Perform Feature Selection
                features_to_keep_n = accuracy_tos(df_filtered_decision_scores, df_filtered_decision_function, df_filtered_training_prediction, 
                                                df_filtered_testing_prediction, y_train, y_test, accuracy)

                # Filter Learners
                df_filtered_decision_scores = df_filtered_decision_scores[features_to_keep_n]
                df_filtered_decision_function = df_filtered_decision_function[features_to_keep_n]
                df_filtered_training_prediction = df_filtered_training_prediction[features_to_keep_n]
                df_filtered_testing_prediction = df_filtered_testing_prediction[features_to_keep_n]
                
                # Build each dataset
                X_comb_training = X_train.join(df_filtered_decision_scores, how='outer')
                X_comb_testing = X_test.join(df_filtered_decision_function, how='outer')
            
            # Fix datasets depending of number of stacking layers
            if n_layers == 0:
                X_comb_training_filtered = X_train.join(df_filtered_decision_scores, how='outer')
                X_comb_testing_filtered = X_test.join(df_filtered_decision_function, how='outer')
            else:
                X_comb_training = X_train.join(df_decision_scores, how='outer')
                X_comb_testing = X_test.join(df_decision_function, how='outer')
                X_comb_training_filtered = X_train.join(df_filtered_decision_scores, how='outer')
                X_comb_testing_filtered = X_test.join(df_filtered_decision_function, how='outer')
            
            # Build Dataset Dictionary
            if mode == 'stacking-decisions': 
                X_dict = {mode: (df_decision_scores, df_decision_function)}
            elif mode == 'stacking-decisions-filtered':
                X_dict = {mode: (df_filtered_decision_scores, df_filtered_decision_function)}
            elif mode == 'stacking-decisions-combined':
                X_dict = {mode: (X_comb_training, X_comb_testing)}
            else:
                X_dict = {mode: (X_comb_training_filtered, X_comb_testing_filtered)}
                
            # Evaluate Meta-learner over each dataset
            (_, _, _, _, pred_tuple, raw_decision_tuple) = evaluate_metalearner(metalearner, X_dict, y_train, y_test, threshold)
            
        else:
            print('You need to define a dictionary of representation learners')
            pred_tuple = None
            
    return pred_tuple, X_dict


def transcation_report(metalearner, X, y, month_indicator, channel_indicator, type_indicator, test_size, 
                       threshold=0.99, accuracy='precision', representation_learning_model_dict=None, 
                       mode='stacking-filtered', n_layers=0, path=None):
    
    """
    Generates reports regarding the quantity of transactions blocked by type and by channel and 
    compares it with the real amount. Plots the percentage of transactions that are blocked per month.
    
    Args:
        model (Object): anomaly detector or classifier.
        X (Dataframe): dataset (independet variables).
        y (Series): independent variables.
        k (int): number of folds for cross validatioon.
        metalearner_name (String): name of learner.
        threshold (float): threshold that states where to regard an observation as an anomaly (in [0,1]).
        accuracy (String): accuracy metric to evaluate learners.
        representation_learning_model_dict (Dictionary): representation learners dicionary.
        rep_learner_codes (String): representation learner codes,
        mode (String): cross validation framework.
        n_layers (int): number of additional layers of stacking.
        path (String): path to save figures and results.
        
    Return:
        df_channel_fraud (Dataframe): Report of number of blocked transactions by channel.
        df_type_fraud (Dataframe): Report of number of blocked transactions by type.
    """
    
    # Split the dataset into training and testing sets
    cut = int(np.ceil(X.shape[0]*(1 - test_size)))
    X_train = X.iloc[:cut, :]; X_test = X.iloc[cut:, :]
    y_train = y.iloc[:cut]; y_test = y.iloc[cut:]
    month_train = month_indicator.iloc[:cut]; month_test = month_indicator.iloc[cut:]
    
    # Make prediction with classifier
    pred_tuple, _ = make_prediction(metalearner, (X_train, X_test), (y_train, y_test), threshold, accuracy, 
                        representation_learning_model_dict, mode, n_layers)
    
    # Unpack prediction tuple
    (y_pred_train, y_pred_test) = pred_tuple
    
    # Append month indicator to predictions, group them by month, and get percentage of how many
    # transactions are blocked by month
    y_pred_train['month'] = month_train
    y_pred_test['month'] = month_test
    y_pred_total = y_pred_train.append(y_pred_test)
    total_transactions_per_month = y_pred_total.groupby('month').count()[mode]
    blocked_transactions_per_month = y_pred_total.groupby('month').sum()[mode]
    percentage_blocked_transactions_per_month = blocked_transactions_per_month/total_transactions_per_month
    
    # Use same methodology to get the real percentage of fraudulent montly transactions
    y = y.to_frame()
    y['month'] = month_indicator
    fraud_transactions_per_month = y.groupby('month').sum()['fraud']
    percentage_fraud_transactions_per_month = fraud_transactions_per_month/total_transactions_per_month
    
    intersection_month = month_indicator.iloc[cut]
    
    # Get how many frauds where identified by channel and by transaction_mode and contrast it
    # with the real statistics
    y_pred_total['channel'] = channel_indicator; y['channel'] = channel_indicator
    y_pred_total['transaction_mode'] = type_indicator; y['transaction_mode'] = type_indicator 
    y_pred_channel_blocked = y_pred_total[['channel', mode]].groupby('channel').sum()[mode]
    y_channel_fraud = y[['channel', 'fraud']].groupby('channel').sum()['fraud']
    y_pred_type_blocked = y_pred_total[['transaction_mode', mode]].groupby('transaction_mode').sum()[mode]
    y_type_fraud = y[['transaction_mode', 'fraud']].groupby('transaction_mode').sum()['fraud']
    y_percentage_channel_blocked = y_pred_channel_blocked/y_channel_fraud
    y_percentage_type_blocked = y_pred_type_blocked/y_type_fraud
    df_channel_fraud = pd.DataFrame({'Predicted':y_pred_channel_blocked, 'Real': y_channel_fraud,
                                     'Rate': y_percentage_channel_blocked})
    df_type_fraud = pd.DataFrame({'Predicted':y_pred_type_blocked, 'Real': y_type_fraud,
                                     'Rate': y_percentage_type_blocked})
    
    # Plot
    t = list(range(1,13))
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, percentage_blocked_transactions_per_month, color='navy', label='Precentage of Blocked Transactions')
    ax.scatter(t, percentage_blocked_transactions_per_month, marker='o', color='red')
    ax.plot(t, percentage_fraud_transactions_per_month, color='red', label='Precentage of Fraudulent Transactions')
    ax.scatter(t, percentage_fraud_transactions_per_month, marker='o', color='red')
    ax.axvline(intersection_month, color='lime')
    ax.legend(loc='best')
    ax.set_xlabel('month')
    ax.set_title('Model Blocked transactions per month')
    
    if path:
        df_channel_fraud.to_csv(os.path.join(path,'channel_report.csv'))
        df_type_fraud.to_csv(os.path.join(path,'transaction_mode_report.csv'))
        fig.savefig(os.path.join(path,'monthly_blocked_transactions.png'), dpi=100)
    
    return df_channel_fraud, df_type_fraud

def plotting_outlier(metalearner, X, y, test_size, threshold=0.99, accuracy='precision', 
                    representation_learning_model_dict=None, mode='stacking-filtered', n_layers=0, path=None):
    
    """
    Plots the dataset by reducing its dimensionality to 2 components using PCA. Tool to visualize outlier.
    
    Args:
        model (Object): anomaly detector or classifier.
        X (Dataframe): dataset (independet variables).
        y (Series): independent variables.
        k (int): number of folds for cross validatioon.
        metalearner_name (String): name of learner.
        threshold (float): threshold that states where to regard an observation as an anomaly (in [0,1]).
        accuracy (String): accuracy metric to evaluate learners.
        representation_learning_model_dict (Dictionary): representation learners dicionary.
        rep_learner_codes (String): representation learner codes,
        mode (String): cross validation framework.
        n_layers (int): number of additional layers of stacking.
        path (String): path to save figures and results.

    Output:
        2D dataset plot differenced by label
    """

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # Make prediction with classifier
    pred_tuple, X_dict = make_prediction(metalearner, (X_train, X_test), (y_train, y_test), threshold, accuracy, 
                        representation_learning_model_dict, mode, n_layers)
            
    # Initalize PCA object and get coordinates
    pca_prince = PCA(n_components=2, n_iter=3, rescale_with_mean=False, rescale_with_std=False)
    coordinates_train = pca_prince.fit_transform(X_dict[mode][0]).astype(float)
    coordinates_train['pred'] = pred_tuple[0].astype('category')
    coordinates_train.columns = ['cord1', 'cord2', 'pred']
    coordinates_test = pca_prince.fit_transform(X_dict[mode][1]).astype(float)
    coordinates_test['pred'] = pred_tuple[1].astype('category')
    coordinates_test.columns = ['cord1', 'cord2', 'pred']
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    sns.scatterplot(data=coordinates_train, x='cord1', y='cord2', hue='pred', ax=ax[0])
    ax[0].set_xlabel('First Component')
    ax[0].set_ylabel('Second Component')
    ax[0].set_title('Training 2 Component PCA Projection')
    sns.scatterplot(data=coordinates_test, x='cord1', y='cord2', hue='pred', ax=ax[1])
    ax[1].set_xlabel('First Component')
    ax[1].set_ylabel('Second Component')
    ax[1].set_title('Testing 2 Component PCA Projection')

    if path:
        title = 'oulier_pca_2dim_plot.png'
        fig.savefig(os.path.join(path, title), dpi=100)
            
def plot_roc_curve(true_labels, pred_labels, fig, ax):
    
    """
    Plot ROC Curve.
    
    Args:
        true_labels: true labels.
        pred_labels: predicted labels.
        fig: figure.
        ax: axis.
        
    Output:
        ROC Curve
    """
    
    # Get ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, pred_labels)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color="darkorange", lw=2,
             label="ROC curve (area = %0.2f)" % roc_auc)
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver operating characteristic example")
    ax.legend(loc="lower right")

def copytree(dir, src, dst, symlinks=False, ignore=None):

    """ 
    Used to copy a whole directory into a destination. For colab only.

    Args:
        dir: name of the directory.
        src: source path.
        dst: destination path. 
    """

    dst = os.path.join(dst,dir)

    try:
        os.mkdir(dst)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exist')
        else:
            raise

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)