from imp_NN_training import Train_NN_Policy, General_Train_NN_Policy_parameters
from imp_validate_datasets import validate_datasets_function
from DSL_general_methods import Simple_train
from imp_truck_trailer_multi_stage_DSL import Truck_Trailer_Multi_Stage_Use_case

from DSL_functions import Train_policy, Validate_datasets


def run_DSL_operation():
    use_case = Truck_Trailer_Multi_Stage_Use_case()
    use_case.set_self_parameters()

    output_map = 'DSL_truck_trailer_delete_later'

    general_train_nn_policy_parameters = General_Train_NN_Policy_parameters(do_hypertuning=True, hypertuning_epochs=100, hypertuning_factor=3,
                                                            project_name='hyper_trials', NN_training_epochs=150, hyperparameter_file="hyperparameterfile", use_case=use_case)
    

    train_NN,  validate_datasets, = give_DSL_functions(train_function = Train_NN_Policy, general_train_policy_parameters = general_train_nn_policy_parameters, validation_function = validate_datasets_function, general_validation_parameters = False)
    
    Simple_train(train_NN, validate_datasets, 'truck_trailer_multi_stage_correct_comb_1_2_v1.csv', output_map)


def give_DSL_functions(train_function, general_train_policy_parameters, validation_function, general_validation_parameters):
    train_NN = Train_policy(train_function, general_train_policy_parameters)

    validate_datasets = Validate_datasets(validation_function, general_validation_parameters)
    
    return train_NN, validate_datasets

if __name__ == "__main__":
    run_DSL_operation()