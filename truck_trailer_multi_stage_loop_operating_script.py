import math
import sys

#EDIT 04 JUNE - START
import tensorflow as tf
import random
import numpy as np
#EDIT 04 JUNE - END

from imp_NN_training import Train_NN_Truck_Trailer_Multi_Stage_Loop_Policy, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters, Retrain_NN_Policy, General_Retrain_NN_Policy_parameters, Retrain_NN_Policy_parameters, Train_3_NN_Policy
from imp_validate_datasets import validate_datasets_function, Validation_parameters
from imp_validate_trace_datasets import validate_trace_datasets_function, Validation_trace_parameters
from DSL_general_methods import Dagger, NDI, CL, Simple_train
from Use_Case_class import Use_Case
from imp_truck_trailer_multi_stage_loop_DSL import Truck_Trailer_Multi_Stage_Loop_Use_case, standalone_simulate_function, standalone_simulate_function_nudge, initialize_ocp, General_Simulation_parameters
from paper_scripts.random_tree import Train_Random_Tree_Policy, General_Train_Random_Tree_Policy_Parameters

from DSL_data_classes import DSL_Data_Point, DSL_Data_Set, DSL_Trace_Data_Set
from DSL_policy_case import Policy
from DSL_functions import Train_policy, Simulate_system, Simulate_system_traces, Validate_datasets, Validate_trace_datasets

from simulator import simulator_omega_init

def run_DSL_operation_simpletrain():
    use_case = Truck_Trailer_Multi_Stage_Loop_Use_case()
    use_case.set_self_parameters()

    output_map = 'DSL_truck_trailer_multi_stage_loop_split_models_model_3_small_notrelu'
    #[solve, Sim_cart_pole_dyn] = initialize_ocp(0, 0, 0, 0)

    general_train_nn_policy_parameters = General_Train_NN_Policy_parameters(do_hypertuning=True, hypertuning_epochs=50, hypertuning_factor=3,
                                                            project_name='hyper_trials', NN_training_epochs=100, hyperparameter_file="hyperparameterfile", use_case=use_case)

    train_NN, validate_datasets = give_DSL_functions_simpletrain(train_function = Train_NN_Truck_Trailer_Multi_Stage_Loop_Policy, general_train_policy_parameters = general_train_nn_policy_parameters,
    validation_function = validate_datasets_function, general_validation_parameters = False)
    
    Simple_train(train_NN, validate_datasets, "truck_trailer_multi_stage_loop_traces_index_v1_dataset_model_3.csv", output_map)

def run_DSL_operation():
    use_case = Truck_Trailer_Multi_Stage_Loop_Use_case()
    use_case.set_self_parameters()

    #output_map = 'Without_prune_0p5_relu_correct'
    output_map = 'seed_without_prune_0p5_relu_correct'
    #[solve, Sim_cart_pole_dyn] = initialize_ocp(0, 0, 0, 0)
    simu = simulator_omega_init()
    
    #fixed hypertuning_epochs=50 and NN_training_epochs=100
    general_train_nn_policy_parameters = General_Train_NN_Policy_parameters(do_hypertuning=True, hypertuning_epochs=50, hypertuning_factor=3,
                                                            project_name='hyper_trials', NN_training_epochs=100, hyperparameter_file="hyperparameterfile", use_case=use_case)
    
    general_train_random_tree_policy_parameters = General_Train_Random_Tree_Policy_Parameters(n_estimators=100)
    
    general_retrain_nn_policy_parameters = General_Retrain_NN_Policy_parameters(NN_training_epochs=4, use_case=use_case)
    
    general_simulation_parameters = General_Simulation_parameters(function=simu)

    train_NN, retrain_NN, simulate_system, simulate_system_traces, validate_datasets, validate_trace_datasets, expert_policy = give_DSL_functions(train_function = Train_NN_Truck_Trailer_Multi_Stage_Loop_Policy, general_train_policy_parameters = general_train_nn_policy_parameters,
    retrain_function = Retrain_NN_Policy, general_retrain_policy_parameters = general_retrain_nn_policy_parameters, simulation_function = standalone_simulate_function, general_simulation_parameters = general_simulation_parameters, validation_function = validate_datasets_function, general_validation_parameters = False,
    validation_trace_function = validate_trace_datasets_function, general_validation_trace_parameters = False, expert_policy_function = use_case.expert_output_function)
    

    #EDIT 05 JUNE - START
    #Dagger(train_NN, simulate_system_traces, validate_datasets, validate_trace_datasets, expert_policy, 'truck_trailer_multi_stage_loop_index_dataset.csv', 'truck_trailer_multi_stage_loop_index_start_points.csv', 'truck_trailer_multi_stage_loop_index_traces.csv', 10, 49, 0, output_map)
    if len(sys.argv) < 2:
        raise ValueError("Please provide a seed as a command-line argument.")
    seed = int(sys.argv[1])
    print(f"\n=== Running DAgger with seed {seed} ===\n")
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Make output folder unique per seed
    output_map_seed = output_map + f"_seed{seed}"

    Dagger(train_NN,
       simulate_system_traces,
       validate_datasets,
       validate_trace_datasets,
       expert_policy,
       'truck_trailer_multi_stage_loop_index_dataset.csv',
       'truck_trailer_multi_stage_loop_index_start_points.csv',
       'truck_trailer_multi_stage_loop_index_traces.csv',
       10, 49, 0, output_map_seed)
    #EDIT 05 JUNE - END
    #NDI(train_NN, simulate_system, validate_datasets, validate_trace_datasets, expert_policy, 'MPC_data\data_set_mpc_example_DSL', 'MPC_data\data_set_mpc_example_start_points_DSL', 'MPC_data\data_set_mpc_example_perfect_paths', 20, 3, 100, 0.1, output_map)
    #CL(train_NN, retrain_NN, 'MPC_data\data_set_mpc_example_DSL', output_map)

def give_DSL_functions_simpletrain(train_function, general_train_policy_parameters, validation_function, general_validation_parameters):
    train_NN = Train_policy(train_function, general_train_policy_parameters)

    validate_datasets = Validate_datasets(validation_function, general_validation_parameters)

    return train_NN, validate_datasets
def give_DSL_functions(train_function, general_train_policy_parameters, retrain_function, general_retrain_policy_parameters, simulation_function, general_simulation_parameters, validation_function, general_validation_parameters, validation_trace_function, general_validation_trace_parameters, expert_policy_function):
    train_NN = Train_policy(train_function, general_train_policy_parameters)
    retrain_NN = Train_policy(retrain_function, general_retrain_policy_parameters)

    simulate_system = Simulate_system(simulation_function, general_simulation_parameters)
    simulate_system_traces = Simulate_system_traces(simulation_function, general_simulation_parameters)

    validate_datasets = Validate_datasets(validation_function, general_validation_parameters)
    validate_trace_datasets = Validate_trace_datasets(validation_trace_function, general_validation_trace_parameters)
    expert_policy = Policy(expert_policy_function)
    
    return train_NN, retrain_NN, simulate_system, simulate_system_traces, validate_datasets, validate_trace_datasets, expert_policy


if __name__ == "__main__":
    run_DSL_operation()