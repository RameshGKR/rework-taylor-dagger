import os
import csv
import shutil

from numpy import linspace
import numpy as np
from DSL_data_classes import DSL_Data_Set, DSL_Data_Point
from DSL_functions import Train_policy, Validate_datasets
from DSL_policy_case import Policy

from imp_NN_training import Train_NN_Policy_parameters
from imp_validate_datasets import Validation_parameters
from imp_NN_training import Train_NN_Truck_Trailer_Multi_Stage_Loop_Policy, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters
from imp_truck_trailer_multi_stage_loop_DSL import Truck_Trailer_Multi_Stage_Loop_Use_case
from imp_validate_datasets import validate_datasets_function

def retrain_for_cegar(counter_example_file, amount_interval_points, saved_dataset_csv, output_map, hyperparameterfile):
    #set up for Neural Network training
    os.makedirs(output_map)
    use_case = Truck_Trailer_Multi_Stage_Loop_Use_case()
    use_case.set_self_parameters()

    #if you want to do hypertuning set do_hypertuning = True and hyperparameter_file = "hyperparameterfile"
    general_train_nn_policy_parameters = General_Train_NN_Policy_parameters(do_hypertuning=False, hypertuning_epochs=50, hypertuning_factor=3,
                                                            project_name='hyper_trials', NN_training_epochs=100, hyperparameter_file=hyperparameterfile, use_case=use_case)
    
    train_NN = Train_policy(Train_NN_Truck_Trailer_Multi_Stage_Loop_Policy, general_train_nn_policy_parameters)
    validate_datasets = Validate_datasets(validate_datasets_function, False)
    expert_policy = Policy(use_case.expert_output_function)

    train_NN_policy_parameters = Train_NN_Policy_parameters(output_map=output_map)
    validation_parameters = Validation_parameters(output_map=output_map)

    #get counter example points
    amount_of_input_labels = 5
    amount_of_output_labels = 3
    counter_examples_low, counter_examples_high = load_counterexamples(counter_example_file, amount_of_input_labels + amount_of_output_labels)
    extra_datapoints = generate_extra_from_counterexamples(counter_examples_low, counter_examples_high, amount_interval_points)
    counter_example_dataset = put_counterexamples_in_dataset(extra_datapoints)

    counter_example_dataset.output = expert_policy.give_output(counter_example_dataset.input)
    counter_example_dataset.remove_errors()

    #add counter examples to dataset
    total_dataset = DSL_Data_Set()
    total_dataset.initialize_from_csv(saved_dataset_csv)

    total_dataset.append_dataset(counter_example_dataset)

    #train new NN and validate it
    [train_dataset, test_dataset] = total_dataset.split_dataset([0.8, 0.2])
    trained_policy = train_NN.train_policy(train_dataset, train_NN_policy_parameters)

    policy_results = DSL_Data_Set()
    policy_results.input = test_dataset.input
    policy_results.output = trained_policy.give_output(test_dataset.input)

    validate_datasets.validate_datasets(test_dataset, policy_results, validation_parameters)
    
def load_counterexamples(counter_example_file, amount_of_inputs):
    with open(counter_example_file) as file_name:
        csvreader = csv.reader(file_name)
        counter_examples = []
        for idx, row in enumerate(csvreader):
            if idx !=0:
                if row[0]=="new data":
                    counter_examples.append([float(i) for i in row[1:amount_of_inputs+1]])
    
    counter_examples_low = counter_examples[0]
    counter_examples_high = counter_examples[1]
    return counter_examples_low, counter_examples_high

def generate_extra_from_counterexamples(counter_examples_low, counter_examples_high, amount_interval_points):
    extra_datapoints=[]
    line_extra_datapoints = [[]]

    for idx, counter_example_input_low in enumerate(counter_examples_low):    
        counter_example_input_high = counter_examples_high[idx]
        interval_points = list(linspace(counter_example_input_low, counter_example_input_high, num=amount_interval_points))

        new_extra_datapoints = []

        for data_point in line_extra_datapoints:
            for interval_point in interval_points:
                new_data_point = data_point.copy()
                new_data_point.append(interval_point)

                new_extra_datapoints.append(new_data_point)
            
        line_extra_datapoints = new_extra_datapoints

    for line_datapoint in line_extra_datapoints:
        extra_datapoints.append(line_datapoint)
    
    return extra_datapoints

def put_counterexamples_in_dataset(extra_datapoints):
    counter_example_dataset = DSL_Data_Set()

    for extra_datpoint_list in extra_datapoints:
        extra_datapoint_dict_input = {'index': extra_datpoint_list[0], 'x1': extra_datpoint_list[1], 'y1': extra_datpoint_list[2], 'theta0': extra_datpoint_list[3], 'theta1': extra_datpoint_list[4]}
        extra_datapoint_dict_output = {'v0': extra_datpoint_list[5], 'delta0': extra_datpoint_list[6], 'v1': extra_datpoint_list[7]}

        datapoint = DSL_Data_Point(input=extra_datapoint_dict_input, output=extra_datapoint_dict_output)
        counter_example_dataset.append_datapoint(datapoint)

    return counter_example_dataset


# def generate_control_action_for_datapoints(use_case, data_set):
#     columns = data_set.give_columns(give_input=True, give_output=False)
#     control_variables = use_case.give_expert_actions(columns)
    
#     data_set.load_data_from_col_list(control_variables, load_only_output = True)
#     data_set.delete_False_outputs()
#     return data_set


if __name__ == "__main__":
    retrain_for_cegar(False, False, "truck_trailer_multi_stage_loop_traces_index_v1_dataset.csv", "retrain_cegar_delete_later","DSL_truck_trailer_run_2\output_NN_hypertuning\hyperparameterfile")
  