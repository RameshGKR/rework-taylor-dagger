import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from h5_to_yaml import h5_to_yml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from typing import List
from Use_Case_class import Use_Case, give_list_list
from imp_hyper_functions import hypertune, get_hyperparameters_from_file, save_hyperparametersearch_results, save_hyperparameters_to_file
from dataclasses import dataclass
sys.path.insert(0, 'C:\\Users\\Werk\\DSL')
from DSL_data_classes import DSL_Data_Point, DSL_Data_Set, DSL_Trace_Data_Set
from DSL_policy_case import Policy

@dataclass
class General_Train_NN_Policy_parameters:
    do_hypertuning: bool = False  
    hypertuning_epochs: int = 0
    hypertuning_factor: int = 0
    project_name: str = 'hyper_trials'
    NN_training_epochs: int = 0
    hyperparameter_file: str = ""
    use_case: Use_Case = False

@dataclass
class Train_NN_Policy_parameters:
    output_map: str = ""

@dataclass
class General_Retrain_NN_Policy_parameters:
    NN_training_epochs: int = 0
    use_case: Use_Case = False

@dataclass
class Retrain_NN_Policy_parameters:
    hyperparameter_file: str = ""
    saved_weights: str = ""
    output_map: str = ""

def Train_NN_Policy(input_dataset, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters):
    train_dataset, test_dataset = input_dataset.split_dataset([0.8, 0.2])
    use_case = General_Train_NN_Policy_parameters.use_case

    if General_Train_NN_Policy_parameters.do_hypertuning:
        hyper_model = use_case.give_hypermodel(train_dataset.input_dataframe)
        Hypertune_NN(hyper_model, use_case, train_dataset, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters)

    if General_Train_NN_Policy_parameters.do_hypertuning:
        hyper_param_path = str(os.path.join(Train_NN_Policy_parameters.output_map, "output_NN_hypertuning", General_Train_NN_Policy_parameters.hyperparameter_file))
        hyper_parameters = get_hyperparameters_from_file(hyper_param_path)
        # hyper_parameters = get_hyperparameters_from_file(Train_NN_Policy_parameters.output_map + '\output_NN_hypertuning\\'+ General_Train_NN_Policy_parameters.hyperparameter_file)
    else:
        hyper_parameters = get_hyperparameters_from_file(General_Train_NN_Policy_parameters.hyperparameter_file)

    dnn_model = use_case.give_NN_model(hyper_parameters, train_dataset.input_dataframe)

    history = fit_model(dnn_model, train_dataset, General_Train_NN_Policy_parameters.NN_training_epochs)
    save_training_results(use_case, dnn_model, history, hyper_parameters, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters, "training")

    policy = give_NN_policy(use_case, dnn_model)
    return policy

def Train_3_NN_Policy(input_dataset, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters):
    dataset_1, dataset_2, dataset_3 = split_datasets(input_dataset)
    use_case = General_Train_NN_Policy_parameters.use_case

    dnn_model_1 = split_model_training(dataset_1, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters, 1)
    dnn_model_2 = split_model_training(dataset_2, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters, 2)
    dnn_model_3 = split_model_training(dataset_3, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters, 3)

    policy = give_split_NN_policy(use_case, dnn_model_1, dnn_model_2, dnn_model_3)
    return policy

def split_model_training(dataset, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters_in, number):
    dataset.input_dataframe=dataset.input_dataframe.drop(['index'], axis=1)
    train_dataset = dataset
    #train_dataset, test_dataset = dataset.split_dataset([0.8, 0.2])
    use_case = General_Train_NN_Policy_parameters.use_case

    output_map_path = str(os.path.join(Train_NN_Policy_parameters_in.output_map, "model_" + str(number)))
    train_NN_policy_parameters_split = Train_NN_Policy_parameters(output_map=output_map_path)
    # train_NN_policy_parameters_split = Train_NN_Policy_parameters(output_map=Train_NN_Policy_parameters_in.output_map + "\\model_" + str(number))

    if General_Train_NN_Policy_parameters.do_hypertuning:
        hyper_model = use_case.give_hypermodel(train_dataset.input_dataframe)
        Hypertune_NN(hyper_model, use_case, train_dataset, General_Train_NN_Policy_parameters, train_NN_policy_parameters_split)

    if General_Train_NN_Policy_parameters.do_hypertuning:
        file_path = str(os.path.join(train_NN_policy_parameters_split.output_map, "output_NN_hypertuning", General_Train_NN_Policy_parameters.hyperparameter_file))
        hyper_parameters = get_hyperparameters_from_file(file_path)
        # hyper_parameters = get_hyperparameters_from_file(train_NN_policy_parameters_split.output_map+ '\output_NN_hypertuning\\'+ General_Train_NN_Policy_parameters.hyperparameter_file)
    else:
        hyper_parameters = get_hyperparameters_from_file(General_Train_NN_Policy_parameters.hyperparameter_file)

    dnn_model = use_case.give_NN_model(hyper_parameters, train_dataset.input_dataframe)

    history = fit_model(dnn_model, train_dataset, General_Train_NN_Policy_parameters.NN_training_epochs)
    save_training_results(use_case, dnn_model, history, hyper_parameters, General_Train_NN_Policy_parameters, train_NN_policy_parameters_split,"training")

    return dnn_model

def split_datasets(input_dataset):
    dataset_1 = DSL_Data_Set()
    dataset_2 = DSL_Data_Set()
    dataset_3 = DSL_Data_Set()

    for datapoint in input_dataset:
        index_datapoint = datapoint.input_dataframe['index'][0]
        if index_datapoint<20:
            dataset_1.append_datapoint(datapoint)
        elif index_datapoint>19 and index_datapoint<30:
            dataset_2.append_datapoint(datapoint)
        else:
            dataset_3.append_datapoint(datapoint)
    
    return dataset_1, dataset_2, dataset_3

def give_split_NN_policy(use_case, dnn_model_1, dnn_model_2, dnn_model_3):
    def get_control_action_from_NN(input_datapoint):
        """This function gives a control action from the NN"""
        # deinput=input_datapoint.input_dataframe
        # deinputnumpy=input_datapoint.input_dataframe.to_numpy()
        # deinputnumpysplit=input_datapoint.input_dataframe.to_numpy()[0]
        # deinputtensor=tf.convert_to_tensor(input_datapoint.input_dataframe)

        index_datapoint = input_datapoint.input_dataframe['index'][0]
        input_datapoint.input_dataframe=input_datapoint.input_dataframe.drop(['index'], axis=1)
        if index_datapoint<20:
            NN_output = dnn_model_1(tf.convert_to_tensor(input_datapoint.input_dataframe))
        elif index_datapoint>19 and index_datapoint<30:
            NN_output = dnn_model_2(tf.convert_to_tensor(input_datapoint.input_dataframe))
        else:
            NN_output = dnn_model_3(tf.convert_to_tensor(input_datapoint.input_dataframe))

        NN_output_dict = {}

        if len(use_case.labels_output) == 1:
            NN_output_dict[use_case.labels_output[0]] = NN_output.numpy()[0][0]
        else:
            for idx, NN_output_parameter in enumerate(NN_output.numpy()[0]):
                NN_output_dict[use_case.labels_output[idx]] = NN_output_parameter

        output_datapoint = DSL_Data_Point(output = NN_output_dict)

        return output_datapoint
    
    return Policy(get_control_action_from_NN)

def Train_NN_Truck_Trailer_Multi_Stage_Loop_Policy(input_dataset, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters):
    #remove indexs from dataset
    input_dataset.input_dataframe=input_dataset.input_dataframe.drop(['index'], axis=1)
    train_dataset=input_dataset
    #train_dataset, test_dataset = input_dataset.split_dataset([0.8, 0.2])
    use_case = General_Train_NN_Policy_parameters.use_case

    if General_Train_NN_Policy_parameters.do_hypertuning:
        hyper_model = use_case.give_hypermodel(train_dataset.input_dataframe)
        Hypertune_NN(hyper_model, use_case, train_dataset, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters)
    
    if General_Train_NN_Policy_parameters.do_hypertuning:
        # hyper_parameters = get_hyperparameters_from_file(Train_NN_Policy_parameters.output_map + '\output_NN_hypertuning\\'+ General_Train_NN_Policy_parameters.hyperparameter_file)
        hyper_parameters = get_hyperparameters_from_file(
            str(os.path.join(Train_NN_Policy_parameters.output_map, 'output_NN_hypertuning', General_Train_NN_Policy_parameters.hyperparameter_file))
        )
    else:
        hyper_parameters = get_hyperparameters_from_file(General_Train_NN_Policy_parameters.hyperparameter_file)
        
    dnn_model = use_case.give_NN_model(hyper_parameters, train_dataset.input_dataframe)

    history = fit_model(dnn_model, train_dataset, General_Train_NN_Policy_parameters.NN_training_epochs)
    save_training_results(use_case, dnn_model, history, hyper_parameters, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters, "training")

    policy = give_NN_Truck_Trailer_Multi_Stage_policy(use_case, dnn_model)
    return policy

def Hypertune_NN(hyper_model, use_case, train_dataset, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters):
    output_map_name = str(os.path.join(Train_NN_Policy_parameters.output_map, "output_NN_hypertuning", General_Train_NN_Policy_parameters.hyperparameter_file))
    # output_map_name = Train_NN_Policy_parameters.output_map + '\output_NN_hypertuning\\'+ General_Train_NN_Policy_parameters.hyperparameter_file
    os.makedirs(str(os.path.join(Train_NN_Policy_parameters.output_map, "output_NN_hypertuning")))
    # os.makedirs(Train_NN_Policy_parameters.output_map + '\output_NN_hypertuning\\')
    best_hps = hypertune(hyper_model, train_dataset, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters)
   
    save_hyperparametersearch_results(best_hps, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters, use_case.hyperparameters)
    save_hyperparameters_to_file(use_case, best_hps, output_map_name)

def save_training_results(use_case, model, history, hyper_parameters, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters, output_map_label):
    output_map_name = str(os.path.join(Train_NN_Policy_parameters.output_map, 'output_NN_'+ output_map_label))
    # output_map_name = Train_NN_Policy_parameters.output_map + '\output_NN_'+ output_map_label
    os.makedirs(output_map_name)

    plot_loss(history, output_map_name)
    save_model(use_case, model, output_map_name)
    write_output_file(model, output_map_name, General_Train_NN_Policy_parameters, hyper_parameters, use_case.hyperparameters)

def Retrain_NN_Policy(input_dataset, General_Retrain_NN_Policy_parameters, Retrain_NN_Policy_parameters):
    """Retrain a NN"""
    use_case = General_Retrain_NN_Policy_parameters.use_case
    train_dataset, test_dataset = input_dataset.split_dataset([0.8, 0.2])

    hyper_parameters = get_hyperparameters_from_file(Retrain_NN_Policy_parameters.hyperparameter_file)
    saved_model= load_NN_from_weights(use_case, hyper_parameters, train_dataset.input_dataframe, Retrain_NN_Policy_parameters.saved_weights)

    history = fit_model(saved_model, train_dataset, General_Retrain_NN_Policy_parameters.NN_training_epochs)
    save_training_results(use_case, saved_model, history, hyper_parameters, General_Retrain_NN_Policy_parameters, Retrain_NN_Policy_parameters, "training")

    policy = give_NN_policy(use_case, saved_model)
    return policy


def fit_model(dnn_model: tf.keras.Sequential, train_dataset, Trained_epochs: int) -> keras.callbacks.History:
    """This function fit a model"""
    history = dnn_model.fit(
        train_dataset.input_dataframe, train_dataset.output_dataframe,
        validation_split=0.2,
        verbose=0, epochs=Trained_epochs)
    return history

def plot_loss(history: keras.callbacks.History, map_name: str):
    """This function makes and saves the figure with the loss and validation loss"""
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error  [U1,U2]')
    plt.legend()
    plt.grid(True)
    plt.savefig(map_name + '\Figure_loss_and_val_loss_')
    plt.clf()

def save_model(use_case: Use_Case, dnn_model: tf.keras.Sequential, map_name: str):
    """This function saves the model, the weights and saves it as a yaml."""
    dnn_model.save(map_name+'\dnn_model')
    dnn_model.save_weights(map_name+'\dnn_model' + 'weigths.h5')
    h5_to_yml([map_name+'\dnn_model', map_name + '\dnn_model' +'_yaml'], use_case.custom_objects)

def write_output_file(dnn_model: tf.keras.Sequential, output_map_name, General_Train_NN_Policy_parameters, hyper_parameters: List, hyperparameters_labels: List[str]):
    """This function saves the parameters of NN training and the test results"""
    outputfile= open(output_map_name + '\\' + '_outputfile.txt', "a")
    for idx, hyperparameter_label in enumerate(hyperparameters_labels):
        outputfile.write(f"{hyperparameter_label}: {hyper_parameters[idx]}\n")
    outputfile.write(f"The amount of trained epochs are: {General_Train_NN_Policy_parameters.NN_training_epochs}\n")
    outputfile.write(f"The loss function is: {dnn_model.loss}\n")
    dnn_model.summary(print_fn=lambda x: outputfile.write(x + '\n'))
    outputfile.close()

def load_NN_from_weights(use_case: Use_Case, hyper_parameters: List, train_features: pd.DataFrame, weights_file: str) -> tf.keras.Sequential:
    """This function loads a NN from saved weights"""
    NN = use_case.give_NN_model(hyper_parameters, train_features)
    NN.load_weights(weights_file)
    return NN

def give_NN_policy(use_case: Use_Case, NN: tf.keras.Sequential):
    def get_control_action_from_NN(input_datapoint):
        """This function gives a control action from the NN"""
        # deinput=input_datapoint.input_dataframe
        # deinputnumpy=input_datapoint.input_dataframe.to_numpy()
        # deinputnumpysplit=input_datapoint.input_dataframe.to_numpy()[0]
        # deinputtensor=tf.convert_to_tensor(input_datapoint.input_dataframe)
        NN_output = NN(tf.convert_to_tensor(input_datapoint.input_dataframe))

        NN_output_dict = {}

        if len(use_case.labels_output) == 1:
            NN_output_dict[use_case.labels_output[0]] = NN_output.numpy()[0][0]
        else:
            for idx, NN_output_parameter in enumerate(NN_output.numpy()[0]):
                NN_output_dict[use_case.labels_output[idx]] = NN_output_parameter

        output_datapoint = DSL_Data_Point(output = NN_output_dict)

        return output_datapoint
    
    return Policy(get_control_action_from_NN)

def give_NN_Truck_Trailer_Multi_Stage_policy(use_case: Use_Case, NN: tf.keras.Sequential):
    def get_control_action_from_NN(input_datapoint):
        """This function gives a control action from the NN"""
        
        input_datapoint.input_dataframe=input_datapoint.input_dataframe.drop(['index'], axis=1)

        NN_output = NN(tf.convert_to_tensor(input_datapoint.input_dataframe))

        NN_output_dict = {}

        if len(use_case.labels_output) == 1:
            NN_output_dict[use_case.labels_output[0]] = NN_output.numpy()[0][0]
        else:
            for idx, NN_output_parameter in enumerate(NN_output.numpy()[0]):
                NN_output_dict[use_case.labels_output[idx]] = NN_output_parameter

        output_datapoint = DSL_Data_Point(output = NN_output_dict)

        return output_datapoint
    
    return Policy(get_control_action_from_NN)
if __name__ == "__main__":
  
  """## A DNN regression
  """

  Trained_epochs = 400

  name = 'V_mp_27_' + str(Trained_epochs) + '_epochs'
  map_name = 'dnns_motion_planning_example' + '\\' + name

  os.makedirs(map_name)

  DNN_units = 64
  DNN_activation = 'sigmoid'
  DNN_Learning_rate = 0.0112955831
  #do_a_NN_training('data_set_car_example.csv', map_name, name, Trained_epochs, DNN_units, DNN_activation, DNN_Learning_rate)