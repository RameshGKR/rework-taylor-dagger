import os
import math
import numpy as np
import pandas as pd
from DSL_data_classes import DSL_Data_Point, DSL_Data_Set, DSL_Trace_Data_Set
from DSL_policy_case import Policy

from imp_NN_training import Train_NN_Policy_parameters, Retrain_NN_Policy_parameters
from imp_validate_datasets import Validation_parameters
from imp_validate_trace_datasets import Validation_trace_parameters
from imp_truck_trailer_multi_stage_DSL import Truck_Trailer_Multi_Stage_Use_case

def Simple_train(train_NN, validate_datasets, start_dataset_csv, output_map):
    total_dataset = DSL_Data_Set()
    total_dataset.initialize_from_csv(start_dataset_csv)

    os.makedirs(output_map)

    train_NN_policy_parameters = Train_NN_Policy_parameters(output_map=output_map)
    validation_parameters = Validation_parameters(output_map=output_map)

    [train_dataset, test_dataset] = total_dataset.split_dataset([0.8, 0.2])
    trained_policy = train_NN.train_policy(train_dataset, train_NN_policy_parameters)

    policy_results = DSL_Data_Set()
    policy_results.input = test_dataset.input
    policy_results.output = trained_policy.give_output(test_dataset.input)

    validate_datasets.validate_datasets(test_dataset, policy_results, validation_parameters)

def Dagger(train_NN, simulate_system_traces, validate_datasets, validate_trace_datasets, expert_policy, start_dataset_csv, start_point_dataset_csv, expert_trace_dataset_csv, Dagger_loops, trace_length, p, output_map):    
    total_dataset = DSL_Data_Set()
    total_dataset.initialize_from_csv(start_dataset_csv)

    start_point_dataset = DSL_Data_Set()
    start_point_dataset.initialize_from_csv(start_point_dataset_csv)

    expert_trace_dataset = DSL_Trace_Data_Set()
    expert_trace_dataset.initialize_from_csv(expert_trace_dataset_csv)

    for idx in range(Dagger_loops):
        # idx = idx+12
        iteration_output_map = str(os.path.join(output_map, "iteration_"+str(idx+1)))
        os.makedirs(iteration_output_map)

        train_NN_policy_parameters = Train_NN_Policy_parameters(output_map=iteration_output_map)
        validation_parameters = Validation_parameters(output_map=iteration_output_map)
        validation_trace_parameters = Validation_trace_parameters(output_map=iteration_output_map)

        [train_dataset, test_dataset] = total_dataset.split_dataset([0.8, 0.2])
        # EDIT START- 19-JUNE-2025
        # We need train_dataset to compare training data across dagger iterations.
        train_dataset.write_dataset_to_csv(os.path.join(iteration_output_map, "train_dataset.csv"))
        # EDIT END - 19-JUNE-2025
        trained_policy = train_NN.train_policy(train_dataset, train_NN_policy_parameters)

        policy_results = DSL_Data_Set()
        policy_results.input = test_dataset.input
        policy_results.output = trained_policy.give_output(test_dataset.input)

        validate_datasets.validate_datasets(test_dataset, policy_results, validation_parameters)

        beta = math.pow(p, idx)
        #loop_policy = expert_policy.statistical_combination([trained_policy],[beta,(1-beta)])

        #loop_policy = expert_policy

        trace_dataset = simulate_system_traces.simulate_system_traces(trained_policy, start_point_dataset.input, trace_length)
        print_dataset = DSL_Data_Set()
        print_dataset.append_trace_dataset(trace_dataset)
        print_dataset.write_dataset_to_csv(iteration_output_map+"\\all_traces_dataset.csv")
        
        # Commented to have plots without pruning for ts=0.5
        trace_dataset = compare_traces(trace_dataset, expert_trace_dataset, 0.1)

        print_dataset = DSL_Data_Set()
        print_dataset.append_trace_dataset(trace_dataset)
        print_dataset.write_dataset_to_csv(iteration_output_map+"\\pruned_traces_dataset.csv")
       
        #validate_trace_datasets.validate_trace_datasets(expert_trace_dataset, trace_dataset, validation_trace_parameters)

        loop_dataset = DSL_Data_Set()
        loop_dataset.append_trace_dataset(trace_dataset)
        loop_dataset.output = expert_policy.give_output(loop_dataset.input)
        loop_dataset.remove_errors()

        total_dataset.append_dataset(loop_dataset)
        total_dataset.write_dataset_to_csv(iteration_output_map+"\\dataset.csv")

def compare_traces(NN_trace_dataset, expert_trace_dataset, limit):
    for dataset_index in range(len(expert_trace_dataset.datasets)):
        remove_indexes = []
        for state_index in range(len(NN_trace_dataset.datasets[dataset_index].input_dataframe["x1"])):
            expert_x1 = expert_trace_dataset.datasets[dataset_index].input_dataframe["x1"][state_index]
            expert_y1 = expert_trace_dataset.datasets[dataset_index].input_dataframe["y1"][state_index]
            NN_x1 = NN_trace_dataset.datasets[dataset_index].input_dataframe["x1"][state_index]
            NN_y1 = NN_trace_dataset.datasets[dataset_index].input_dataframe["y1"][state_index]

            distance= np.sqrt(np.square(np.array(expert_x1)-np.array(NN_x1))+np.square(np.array(expert_y1)-np.array(NN_y1)))

            if distance >= limit:
                remove_indexes.append(state_index)
        
        temp_dataset = NN_trace_dataset.datasets[dataset_index]
        temp_dataset.is_trace = False
        temp_dataset.remove_datapoints(remove_indexes)
        temp_dataset.is_trace = True

        NN_trace_dataset.datasets[dataset_index] = temp_dataset
    return NN_trace_dataset

def NDI(train_NN, simulate_system, validate_datasets, validate_trace_datasets, expert_policy, start_dataset_csv, start_point_dataset_csv, expert_trace_dataset_csv, number_of_paths, max_iteration, trace_length, max_diff, output_map):
    start_point_dataset = DSL_Data_Set()
    start_point_dataset.initialize_from_csv(start_point_dataset_csv)

    first_policy_dataset = DSL_Data_Set()
    first_policy_dataset.initialize_from_csv(start_point_dataset_csv)

    iteration_output_map = output_map+"\iteration_"+str(0)
    train_NN_policy_parameters = Train_NN_Policy_parameters(output_map=iteration_output_map)
    policy = train_NN.train_policy(first_policy_dataset, train_NN_policy_parameters)

    total_traces = DSL_Trace_Data_Set()
    policy_array =[policy]

    for i in range(max_iteration):
        iteration_output_map = output_map+"\iteration_"+str(i+1)
        train_NN_policy_parameters = Train_NN_Policy_parameters(output_map=iteration_output_map)

        total_traces = get_start_points(start_point_dataset, number_of_paths, expert_policy, total_traces)

        loop_traces = DSL_Trace_Data_Set()
        training_dataset = DSL_Data_Set()

        for trace in total_traces:
            loop_trace = ExteND(expert_policy, policy, trace, trace_length, max_diff, simulate_system)
            training_dataset.append_dataset(loop_trace)
            
            if loop_trace.length>trace.length:
                loop_traces.append_dataset(loop_trace)
            else:
                loop_traces.append_dataset(trace)

        policy = train_NN.train_policy(training_dataset, train_NN_policy_parameters)
        policy_array.append(policy)
        total_traces = loop_traces
    
def get_start_points(start_point_dataset, number_of_paths, expert_policy, total_traces):
    start_point_dataset.shuffle()

    for idx in range(number_of_paths):
        start_point = start_point_dataset.get_iter_datapoint_from_dataset(idx)

        k_start_points_dataset = DSL_Data_Set(is_trace=True)
        k_start_points_dataset.append_datapoint(start_point)
        k_start_points_dataset.output = expert_policy.give_output(k_start_points_dataset.input)

        total_traces.append_dataset(k_start_points_dataset)
  
    return total_traces

def ExteND(expert_policy, policy, trace, extend_end_index, max_diff, simulate_system):
    extend_trace = DSL_Data_Set(is_trace = True)

    for n in range(extend_end_index):
        if n+1>trace.length:
            loop_data_point = extend_trace.get_iter_datapoint_from_dataset(n-1)
            
            trace_datapoint = simulate_system.simulate_system(loop_data_point)
            output_trace_datapoint = expert_policy.give_output(trace_datapoint)
            trace_datapoint.set_datapoint_output_with_dataframe(output_trace_datapoint.output_dataframe)
        else:
            trace_datapoint = trace.get_iter_datapoint_from_dataset(n)
        
        extend_trace.append_datapoint(trace_datapoint)

        policy_input_datapoint = DSL_Data_Point()
        policy_input_datapoint.set_datapoint_input_with_dataframe(trace_datapoint.input_dataframe)

        policy_output_datapoint = policy.give_output(policy_input_datapoint)

        if check_output(trace_datapoint.output_dataframe, policy_output_datapoint.output_dataframe, max_diff):
            break
    
    return extend_trace

def check_output(expertpolicy_dataframe, policy_dataframe, max_diff):
    totalbool = True

    for output_label in expertpolicy_dataframe.columns:
        bool=expertpolicy_dataframe[output_label][0]-max_diff<=policy_dataframe[output_label][0]<=expertpolicy_dataframe[output_label][0]+max_diff
        totalbool=bool and totalbool

    return totalbool

def CL(train_NN, retrain_NN, start_dataset_csv, output_map):
    total_dataset = DSL_Data_Set()
    total_dataset.initialize_from_csv(start_dataset_csv)

    dataset_list = sort_train_dataset(total_dataset, 9)

    training_dataset = DSL_Data_Set()

    for idx, dataset in enumerate(dataset_list):
        iteration_output_map = output_map+"\iteration_"+str(idx+1)
        
        training_dataset.append_dataset(dataset)
        [train_dataset, test_dataset] = training_dataset.split_dataset([0.8, 0.2])
        
        if idx == 0:
            train_NN_policy_parameters = Train_NN_Policy_parameters(output_map=iteration_output_map)
            policy = train_NN.train_policy(train_dataset, train_NN_policy_parameters)
        else:
            retrain_NN_Policy_parameters = Retrain_NN_Policy_parameters(hyperparameter_file=output_map+'\iteration_1\output_NN_hypertuning\hyperparameterfile', saved_weights=output_map+"\iteration_"+str(idx)+'\output_NN_training\dnn_modelweigths.h5', output_map=iteration_output_map)
            policy = retrain_NN.train_policy(train_dataset, retrain_NN_Policy_parameters)

def sort_train_dataset(total_dataset, amount_of_datasets):
    dataset_list = []
    for _ in range(amount_of_datasets):
        dataset = DSL_Data_Set()
        dataset_list.append(dataset)

    sort_pandaframe = pd.DataFrame(columns=['abs_pos','datapoint'])

    for idx, datapoint in enumerate(total_dataset):
        pos = datapoint.input_dataframe['POS'][0]
        abs_pos = np.abs(pos)

        sort_pandaframe.loc[idx] = [abs_pos, datapoint]

    sort_pandaframe = sort_pandaframe.sort_values(by=['abs_pos'])
    length = sort_pandaframe.shape[0]
    amount = int(np.ceil(length/amount_of_datasets))
    begin_index = 0

    for i in range(amount_of_datasets-1):
        end_index = begin_index + amount
        loop_dataframe = sort_pandaframe.iloc[begin_index:end_index]

        for datapoint in loop_dataframe['datapoint']:
            dataset_list[i].append_datapoint(datapoint)

        begin_index = end_index

    loop_dataframe = sort_pandaframe.iloc[begin_index:]

    for datapoint in loop_dataframe['datapoint']:
        dataset_list[i+1].append_datapoint(datapoint)

    return dataset_list
