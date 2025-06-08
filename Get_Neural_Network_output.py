import os
import sys
import pandas as pd
import tensorflow as tf
import csv
from typing import List
from Use_Case_class import Use_Case
from imp_truck_trailer_multi_stage_DSL import Truck_Trailer_Multi_Stage_Use_case
from imp_validate_datasets import Validation_parameters, validate_datasets_function
from imp_NN_training import load_NN_from_weights, give_NN_policy
from DSL_data_classes import DSL_Data_Point, DSL_Data_Set
from DSL_functions import Validate_datasets

def get_control_action_from_NN(use_case: Use_Case, NN: tf.keras.Sequential,input_datapoint):
    """This function gives a control action from the NN"""

    NN_output = NN(tf.convert_to_tensor(input_datapoint.input_dataframe))

    NN_output_dict = {}

    if len(use_case.labels_output) == 1:
        NN_output_dict[use_case.labels_output[0]] = NN_output.numpy()[0][0]
    else:
        for idx, NN_output_parameter in enumerate(NN_output.numpy()[0]):
            NN_output_dict[use_case.labels_output[idx]] = NN_output_parameter

    output_datapoint = DSL_Data_Point(output = NN_output_dict)

    return output_datapoint

def get_NN_output(hyperparameterfile, datafile, modelweights, input_datapoint):
    use_case = Truck_Trailer_Multi_Stage_Use_case()
    use_case.set_self_parameters()

    with open(hyperparameterfile) as file_name:
        csvreader = csv.reader(file_name)
        hyperparameters = []
        for row in csvreader:
            hyperparameters.append(row[0])
    
    total_dataset = DSL_Data_Set()
    total_dataset.initialize_from_csv(datafile)
    
    NN = load_NN_from_weights(use_case, hyperparameters, total_dataset.input_dataframe, modelweights)

    output_datapoint = get_control_action_from_NN(use_case, NN, input_datapoint)
    
    return output_datapoint

def get_NN_validate_results(hyperparameterfile, datafile, modelweights, output_map):
    use_case = Truck_Trailer_Multi_Stage_Use_case()
    use_case.set_self_parameters()

    with open(hyperparameterfile) as file_name:
        csvreader = csv.reader(file_name)
        hyperparameters = []
        for row in csvreader:
            hyperparameters.append(row[0])
    
    total_dataset = DSL_Data_Set()
    total_dataset.initialize_from_csv(datafile)
    
    os.makedirs(output_map)

    validation_parameters = Validation_parameters(output_map=output_map)

    [train_dataset, test_dataset] = total_dataset.split_dataset([0.8, 0.2])
    
    NN = load_NN_from_weights(use_case, hyperparameters, total_dataset.input_dataframe, modelweights)
    trained_policy = give_NN_policy(use_case, NN)

    policy_results = DSL_Data_Set()
    policy_results.input = test_dataset.input
    policy_results.output = trained_policy.give_output(test_dataset.input)

    validate_datasets = Validate_datasets(validate_datasets_function, validation_parameters)

    validate_datasets.validate_datasets(test_dataset, policy_results, validation_parameters)
    
if __name__ == "__main__":
    hyperparameterfile = str(os.path.join("DSL_truck_trailer_delete_later", "output_NN_hypertuning", "hyperparameterfile"))
    # hyperparameterfile = "DSL_truck_trailer_delete_later\output_NN_hypertuning\hyperparameterfile"
    datafile = "truck_trailer_multi_stage_correct_v1.csv"
    modelweights = str(os.path.join("DSL_truck_trailer_delete_later", "output_NN_training", "dnn_modelweigths.h5"))
    # modelweights = "DSL_truck_trailer_delete_later\output_NN_training\dnn_modelweigths.h5"

    input = [1,2,3,4]  #fill here the datapoint you want get the output from
    input_datapoint = DSL_Data_Point(input={"x1":input[0], "y1":input[1], "theta0":input[2], "theta1":input[3]}) 

    output_datapoint = get_NN_output(hyperparameterfile, datafile, modelweights, input_datapoint)
    get_NN_validate_results(hyperparameterfile, datafile, modelweights, "validate_datasets_results")

    print(output_datapoint.output_dataframe)
