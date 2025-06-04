import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import List
from Use_Case_class import Use_Case, give_list_list
from dataclasses import dataclass

@dataclass
class Validation_parameters:
    output_map:str = ""

def validate_datasets_function(expert_dataset, NN_dataset, General_Validation_parameters, Validation_parameters):
    calculate_and_save_loss(expert_dataset.output_dataframe, NN_dataset.output_dataframe, Validation_parameters.output_map)
    plot_and_save_figures(NN_dataset.output_dataframe, expert_dataset.output_dataframe, Validation_parameters.output_map)

def calculate_and_save_loss(expert_output_dataframe, NN_output_dataframe, output_map):
    mse_loss = np.mean(np.square(np.array(NN_output_dataframe)-np.array(expert_output_dataframe)))
    outputfile= open(output_map + '\\' + 'test_loss.txt', "a")
    outputfile.write(f"The test loss is: {mse_loss}\n")

def plot_and_save_figures(test_predictions: pd.DataFrame, test_labels: pd.DataFrame, map_name: str):
    """This function makes and saves figures and metrics from the NN"""
    error_results_list = []

    for output_label in test_predictions:
        
        plot_predict_vs_real(test_labels, test_predictions, output_label, map_name)
        error_results = plot_histogram(test_labels, test_predictions, output_label, map_name)
        
        error_results_list.append(error_results)

    save_error_metrics(error_results_list, map_name)

def plot_predict_vs_real(test_labels: pd.DataFrame, test_predictions: pd.DataFrame,  output_label: str, map_name: str):
    """This function makes and saves the figure that plots the real vs predicted results"""
    _ = plt.axes(aspect='equal')
    plt.scatter(test_labels[output_label], test_predictions[output_label])
    plt.xlabel('True Values ' + output_label)
    plt.ylabel('Predictions ' + output_label)
    plt.savefig(map_name + '\predict_vs_real_' + output_label )
    plt.clf()

def plot_histogram(test_labels: List[float], test_predictions: List[float], output_label: str, map_name: str) -> List[float]:
    """This function makes and saves the histogram with the faults between the real and the predicted results"""

    error_U = np.array(test_labels[output_label]) - np.array(test_predictions[output_label])
    plt.hist(error_U, bins=25)
    plt.xlabel('Prediction Error ' + output_label)
    _ = plt.ylabel('Count')
    plt.savefig(map_name + '\histogram_' + output_label )
    plt.clf()

    return [np.mean(error_U), np.median(error_U), np.std(error_U)]

def save_error_metrics(error_results_list: List[float], folder: str):
    """This function saves the error metrics in a file"""
    with open(folder+'\Error_results.csv', "a+") as output:
        writer = csv.writer(output, lineterminator='\n')
        for error_results in error_results_list:
            writer.writerow(error_results)

# def get_total_error_results(use_case: Use_Case, dagger_iterations: int, map_name: str):
#     """This function gets the error results for each iteration and saves it in a file"""
#     error_total_results = give_list_list(len(use_case.labels_output)*3)
    
#     for dagger_it in range(dagger_iterations):
#         current_map_name = map_name + '\iteration_' + str(dagger_it)
#         current_map_name=current_map_name+'\output_NN_training\Error_results.csv'
#         with open(current_map_name) as file_name:
#             error_results = np.loadtxt(file_name, delimiter=",")

#         for idx in range(len(use_case.labels_output)*3):
#             error_total_results[idx].append(error_results[idx])

#     save_total_error_results(use_case, error_total_results, map_name)

# def save_total_error_results(use_case: Use_Case, error_total_results: List[List[float]], map_name: str):
#     """This function saves the total error results in a file"""
#     with open(map_name+'\Error_total_results.csv', "a+") as output:
#         writer = csv.writer(output, lineterminator='\n')
#         for idx in range(len(use_case.labels_output)*3):
#             writer.writerow(error_total_results[idx])

