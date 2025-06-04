import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import csv
from typing import List
from Use_Case_class import Use_Case

def hypertune(hyper_model, train_dataset, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters) -> kt.engine.hyperparameters.HyperParameter:
    """This functions hypertunes a model and gives back the best hyperparameters"""
    tuner = kt.Hyperband(hyper_model,
                        objective='val_loss',
                        max_epochs=General_Train_NN_Policy_parameters.hypertuning_epochs,
                        factor=General_Train_NN_Policy_parameters.hypertuning_factor,
                        directory=Train_NN_Policy_parameters.output_map,
                        project_name=General_Train_NN_Policy_parameters.project_name)

    tuner.search(train_dataset.input_dataframe, train_dataset.output_dataframe, validation_split=0.2)

    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    return best_hps

def save_hyperparametersearch_results(best_hps: kt.engine.hyperparameters.HyperParameter, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters, hyperparameters: List[str]):
    """This functions saves the best hyperparameters and the settings of the hyperparametertuning"""
    outputfile= open(Train_NN_Policy_parameters.output_map + '\hyper_outputfile.txt', "a")

    for idx in range(len(best_hps.space)):
        outputfile.write(f"{best_hps.space[idx]}\n")

    outputfile.write(f"The amount epochs are: {General_Train_NN_Policy_parameters.hypertuning_epochs}\n")
    outputfile.write(f"The training factor is: {General_Train_NN_Policy_parameters.hypertuning_factor}\n")
    outputfile.write(f"The best found results are\n")

    for hyperparameter in hyperparameters:
        outputfile.write(f"{hyperparameter}: {best_hps.get(hyperparameter)}\n")

    outputfile.close()

# def get_hyperparameters_from_best_hps(use_case: Use_Case, best_hps: kt.engine.hyperparameters.HyperParameters) -> List:
#     """This function gives back a list with the best hyperparameters from best_hps"""
#     hyperparameters = []
#     for hyperparameter_label in use_case.hyperparameters:
#         hyperparameters.append(best_hps.get(hyperparameter_label))

#     return hyperparameters

def save_hyperparameters_to_file(use_case: Use_Case, best_hps: kt.engine.hyperparameters.HyperParameter, file_name: str):
    """This function saves hyperparameters to a file"""
    outputfile= open(file_name, "w")
    for hyperparameter_label in use_case.hyperparameters:
        outputfile.write(str(best_hps.get(hyperparameter_label))+'\n')
    outputfile.close()
    
def get_hyperparameters_from_file(file_name: str) -> List:
    """This function gives back a list with the best hyperparameters from a file"""
    with open(file_name) as file:
        csvreader = csv.reader(file)
        hyperparameters = []
        for row in csvreader:
            if row[0][0].isdigit():
                hyperparameters.append(float(row[0]))
            else:
                hyperparameters.append(row[0])
    
    return hyperparameters

if __name__ == "__main__":
 
    print('hey')

    
