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
class Validation_trace_parameters:
    output_map:str = ""

def validate_trace_datasets_function(expert_trace_dataset, NN_trace_dataset, General_Validation_trace_parameters, Validation_trace_parameters):
    MSE_result_list = get_MSE(expert_trace_dataset, NN_trace_dataset)
    save_MSE(MSE_result_list, expert_trace_dataset.input_labels, Validation_trace_parameters.output_map)

def get_MSE(expert_trace_dataset, NN_trace_dataset):
    """Calculates the mean square error between the positions and the speeds the NN gets to and the perfect path that the MPC would take. This is saved in a list"""
    MSE_result_list=[]
    
    for input_label in expert_trace_dataset.input_labels:
        MSE_parameter_results = []
        for dataset_index in range(len(expert_trace_dataset.datasets)):
            expert_parameter_trace = expert_trace_dataset.datasets[dataset_index].input_dataframe[input_label]
            NN_parameter_trace = NN_trace_dataset.datasets[dataset_index].input_dataframe[input_label]

            state_parameter_MSE = np.sum(np.square(np.array(expert_parameter_trace)-np.array(NN_parameter_trace)))
            MSE_parameter_results.append(state_parameter_MSE)

        MSE_result_list.append(MSE_parameter_results)
    
    return MSE_result_list

def save_MSE(MSE_result_lists: List[List[float]], labels_input: List[str], output_map: str):
    """Saves the MSE results and plot figures and also saves these"""
    with open(output_map+'\MSE_results.csv', "a+") as output:
        writer = csv.writer(output, lineterminator='\n')
        for idx, MSE_result_list in enumerate(MSE_result_lists):
            writer.writerow(MSE_result_list)
            _ = plt.subplot()
            plt.plot(MSE_result_list)
            plt.savefig(output_map+'\\'+labels_input[idx]+'_MSE_fig')
            plt.clf()

def get_total_MSE_results(use_case: Use_Case, polish_iterations: int, map_name: str):
    """This function gets and saves the MSE results from each iteration"""
    MSE_total_results = give_list_list(len(use_case.labels_input))

    for polish_it in range(polish_iterations):
        current_map_name = map_name + '\iteration_' + str(polish_it)
        current_map_name=current_map_name+'\MSE_results\MSE_results.csv'
        with open(current_map_name) as file_name:
            MSE_results = np.loadtxt(file_name, delimiter=",")

        for idx in range(len(use_case.labels_input)):
            MSE_total_results[idx].append(sum(MSE_results[idx]))
    
    save_total_MSE_results(use_case, map_name, MSE_total_results)

def save_total_MSE_results(use_case: Use_Case, map_name: str, MSE_total_results: List[List[float]]):
    """This function saves the total MSE results"""
    with open(map_name+'\MSE_total_results.csv', "a+") as output:
        writer = csv.writer(output, lineterminator='\n')
        for idx in range(len(use_case.labels_input)):
            writer.writerow(MSE_total_results[idx])
