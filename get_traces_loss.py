import os
import csv
from DSL_data_classes import DSL_Data_Set, DSL_Trace_Data_Set
from imp_NN_training import load_NN_from_weights, give_NN_Truck_Trailer_Multi_Stage_policy, split_datasets, give_split_NN_policy
from DSL_functions import Simulate_system_traces, Validate_trace_datasets
from imp_truck_trailer_multi_stage_loop_DSL import Truck_Trailer_Multi_Stage_Loop_Use_case, standalone_simulate_function, General_Simulation_parameters
from imp_validate_trace_datasets import validate_trace_datasets_function
from simulator import simulator_omega_init
from imp_validate_trace_datasets import Validation_trace_parameters

for i in range(1,13):
	#os.makedirs("model_run_tansig_iteration_"+str(i))
	#hyperparameterfile = "DSL_truck_trailer_multi_stage_split_model_run_tansig\iteration_"+str(i)+"\model_1\output_NN_hypertuning\hyperparameterfile"
	#modelweights = "DSL_truck_trailer_multi_stage_split_model_run_tansig\iteration_"+str(i)+"\model_1\output_NN_training\dnn_modelweigths.h5"
	#datafile_csv = "DSL_truck_trailer_multi_stage_split_model_run_tansig\iteration_"+str(i)+"\dataset.csv"
	#expert_trace_dataset_csv = "truck_trailer_multi_stage_loop_traces_index_v1_traces.csv"
	#start_point_dataset_csv = "truck_trailer_multi_stage_loop_traces_index_v1_start_points.csv"
	#simu = simulator_omega_init()

	main_folder = r"D:\University_Antwerp\Activity_3\DSL\0p5_dsl\trace_loss_0p5_withoutpruned_tanh_correct"
	os.makedirs(main_folder, exist_ok=True)

	iter_folder_name = f"trace_loss_0p5_withoutpruned_tanh_{i}"
	iter_folder_path = os.path.join(main_folder, iter_folder_name)
	os.makedirs(iter_folder_path, exist_ok=True)

	hyperparameterfile = str(os.path.join("Without_prune_0p5_tanh_correct", f"iteration_{i}", "output_NN_hypertuning", "hyperparameterfile"))
	# hyperparameterfile = f"Without_prune_0p5_tanh_correct\\iteration_{i}\\output_NN_hypertuning\\hyperparameterfile"

	modelweights = str(os.path.join("Without_prune_0p5_tanh_correct", f"iteration_{i}", "output_NN_training", "dnn_modelweigths.h5"))
	# modelweights = f"Without_prune_0p5_tanh_correct\\iteration_{i}\\output_NN_training\\dnn_modelweigths.h5"

	datafile_csv = str(os.path.join("Without_prune_0p5_tanh_correct", f"iteration_{i}", "dataset.csv"))
	# datafile_csv = f"Without_prune_0p5_tanh_correct\\iteration_{i}\\dataset.csv"
	expert_trace_dataset_csv = "truck_trailer_multi_stage_loop_index_traces.csv"
	start_point_dataset_csv = "truck_trailer_multi_stage_loop_index_start_points.csv"
	simu = simulator_omega_init()

	#os.makedirs("trace_loss_0p5_withoutprune_relu_"+str(i))
	#hyperparameterfile = "Without_prune_0p5_relu\iteration_"+str(i)+"\output_NN_hypertuning\hyperparameterfile"
	#modelweights = "Without_prune_0p5_relu\iteration_"+str(i)+"\output_NN_training\dnn_modelweigths.h5"
	#datafile_csv = "Without_prune_0p5_relu\iteration_"+str(i)+"\dataset.csv"
	#expert_trace_dataset_csv = "truck_trailer_multi_stage_loop_index_traces.csv"
	#start_point_dataset_csv = "truck_trailer_multi_stage_loop_index_start_points.csv"
	#simu = simulator_omega_init()

	general_simulation_parameters = General_Simulation_parameters(function=simu)
	simulate_system_traces = Simulate_system_traces(standalone_simulate_function, general_simulation_parameters)
	validate_trace_datasets = Validate_trace_datasets(validate_trace_datasets_function, False)

	start_point_dataset = DSL_Data_Set()
	start_point_dataset.initialize_from_csv(start_point_dataset_csv)

	expert_trace_dataset = DSL_Trace_Data_Set()
	expert_trace_dataset.initialize_from_csv(expert_trace_dataset_csv)

	use_case = Truck_Trailer_Multi_Stage_Loop_Use_case()
	use_case.set_self_parameters()

	dataset = DSL_Data_Set()
	dataset.initialize_from_csv(datafile_csv)

	NN_list = []

	with open(hyperparameterfile) as file_name:
		csvreader = csv.reader(file_name)
		hyperparameters = []
		for row in csvreader:
			hyperparameters.append(row[0])

	NN = load_NN_from_weights(use_case, hyperparameters, dataset.input_dataframe, modelweights)
	NN_list.append(NN)

	
	policy = give_NN_Truck_Trailer_Multi_Stage_policy(use_case, NN)
	trace_dataset = simulate_system_traces.simulate_system_traces(policy, start_point_dataset.input, 49)

	#validation_trace_parameters = Validation_trace_parameters(output_map="trace_loss_0p5_withoutprune_relu_"+str(i))
	validation_trace_parameters = Validation_trace_parameters(output_map=iter_folder_path)
	validate_trace_datasets.validate_trace_datasets(expert_trace_dataset, trace_dataset, validation_trace_parameters)