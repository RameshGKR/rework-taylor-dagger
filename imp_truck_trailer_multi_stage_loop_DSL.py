import sys
import os
import csv
from telnetlib import DM
import pandas as pd
from rockit import *
from pylab import *
from casadi import vertcat
import tensorflow as tf
import keras.backend as K
from tensorflow import keras
from keras import layers
from casadi import Function
from rockit import casadi_helpers
import keras_tuner as kt
from typing import List
from dataclasses import dataclass
from Use_Case_class import Use_Case
from Use_Case_class import Data_set
from DSL_data_classes import DSL_Data_Point, DSL_Data_Set, DSL_Trace_Data_Set
from DSL_policy_case import Policy
from DSL_functions import Train_policy, Simulate_system
from simulator import simulator_omega_init
from truck_trailer_multistage_loop_expert_function import get_expert_action_truck_trailer
import random

class Truck_Trailer_Multi_Stage_Loop_Use_case(Use_Case):
    def set_self_parameters(self):
        """This function will set the self parameters"""
        self.labels_input = ['index','x1','y1','theta0','theta1']
        self.NN_labels_input = ['x1','y1','theta0','theta1']
        self.labels_output = ['v0','delta0']
        self.hyperparameters = ['units', 'activation_function', 'learning_rate', 'hidden_layers']
        self.perfect_paths = get_perfect_paths()
        self.end_iterations = [50]*0#len(self.perfect_paths)
        self.timesteps = [0.5]*0#len(self.perfect_paths)
        self.timestep = 0.5
        #self.custom_objects = {'restricted_output_mse': restricted_output_mse}

        #[solve, Sim_cart_pole_dyn] = initialize_ocp(0, 0, 0, 0)
        simu = simulator_omega_init
        self.MPC_function = get_expert_action_truck_trailer
        self.simulation_function = simu
        self.expert_policy = self.expert_output_function
        self.simulate_system = Simulate_system(self.simulate_function)
        
    def give_hypermodel(self, train_features: pd.DataFrame) -> kt.HyperModel:
        """This function gives the hyper model"""
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        hyper_model = Truck_Trailer_Multi_Stage_Loop_HyperModel(normalizer, len(self.NN_labels_input))
        return hyper_model

    def give_NN_model(self, hyper_parameters: List, train_features: pd.DataFrame) -> tf.keras.Sequential:
        """This function gives the NN model"""
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        model = build_and_compile_model(normalizer, hyper_parameters, len(self.NN_labels_input))
        return model
    
    def expert_output_function(self, input_datapoint: DSL_Data_Point) -> DSL_Data_Point:
        MPC_input = vertcat(input_datapoint.input_dataframe['theta1'].iloc[0], input_datapoint.input_dataframe['x1'].iloc[0], input_datapoint.input_dataframe['y1'].iloc[0], input_datapoint.input_dataframe['theta0'].iloc[0])
        try:
            output = self.MPC_function(MPC_input, input_datapoint.input_dataframe['index'].iloc[0])
            output = casadi_helpers.DM2numpy(output, [2,1])
            output_datapoint = DSL_Data_Point(output = {'v0' : output[0], 'delta0' : output[1]})
        except:
            output_datapoint = DSL_Data_Point(output = {'v0' : np.nan,'delta0' : np.nan})
            output_datapoint.error_included = True

        return output_datapoint

    def simulate_function(self, datapoint: DSL_Data_Point, Simulation_parameters) -> DSL_Data_Point:
        """This function gives the next state for a state and control variables"""
        current_X = vertcat(datapoint.input_dataframe['theta1'].iloc[0], datapoint.input_dataframe['x1'].iloc[0], datapoint.input_dataframe['y1'].iloc[0], datapoint.input_dataframe['theta0'].iloc[0])
        current_U = vertcat(datapoint.output_dataframe['delta0'].iloc[0], datapoint.output_dataframe['v0'].iloc[0])

        system_result = self.simulation_function(x0=current_X, u=current_U, T=self.time_step)["xf"]
        system_result=casadi_helpers.DM2numpy(system_result, [2,1])

        index = datapoint.input_dataframe['index'].iloc[0]
        output_datapoint = DSL_Data_Point(input = {'index': index+1, 'x1': system_result[1], 'y1': system_result[2], 'theta0': system_result[3], 'theta1': system_result[0]})

        return output_datapoint
    
    def simulate_function_nudge(self, datapoint: DSL_Data_Point, Simulation_parameters) -> DSL_Data_Point:
        """This function gives the next state for a state and control variables"""
        current_X = vertcat(datapoint.input_dataframe['theta1'].iloc[0], datapoint.input_dataframe['x1'].iloc[0], datapoint.input_dataframe['y1'].iloc[0], datapoint.input_dataframe['theta0'].iloc[0])
        current_U = vertcat(datapoint.output_dataframe['delta0'].iloc[0], datapoint.output_dataframe['v0'].iloc[0])

        system_result = self.simulation_function(x0=current_X, u=current_U, T=self.time_step)["xf"]
        system_result=casadi_helpers.DM2numpy(system_result, [2,1])

        nudge_x1 = random.uniform(-0.01,0.01)
        nudge_y1 = random.uniform(-0.01,0.01)
        nudge_theta0 = random.uniform(-0.01,0.01)
        nudge_theta1 = random.uniform(-0.01,0.01)

        index = datapoint.input_dataframe['index'].iloc[0]
        output_datapoint = DSL_Data_Point(input = {'index': index+1, 'x1': system_result[1]+nudge_x1, 'y1': system_result[2]+nudge_y1, 'theta0': system_result[3]+nudge_theta0, 'theta1': system_result[0]+nudge_theta1})

        return output_datapoint
    # def give_next_state(self, datapoint: List[float], control_variables: List[float], time_step: float, idx: int) -> List[float]:
    #     """This function gives the next state for a state and control variables"""
    #     current_X = vertcat(input[3], input[0], input[1], input[2])
    #     current_U = vertcat(control_variables[0], control_variables[1])

    #     system_result = self.simulation_function(x0=current_X, u=current_U, T=time_step)["xf"]
    #     system_result=casadi_helpers.DM2numpy(system_result, [2,1])

    #     return [system_result[0], system_result[1], system_result[2], system_result[3]]

    # def sample_k_start_points(self, k: int):
    #     """This function gives k start points"""
    #     start_points = []

    #     for _ in range(k):
    #         start_pos = uniform(-0.6, -0.4)

    #         start_point = [start_pos, 0]
    #         start_points.append(start_point)
        
    #     end_iterations = [100]*k
        
    #     return start_points, end_iterations


class Truck_Trailer_Multi_Stage_Loop_HyperModel(kt.HyperModel):
    """This class contains the hypermodel"""
    def __init__(self, norm: tf.keras.layers.experimental.preprocessing.Normalization, amount_of_train_features: int):
        self.normalizer = norm
        self.amount_of_train_features = amount_of_train_features

    def build(self, hp: kt.engine.hyperparameters.HyperParameter) -> tf.keras.Sequential:
        hp_units = hp.Int('units', min_value=16, max_value=96, step=16)
        hp_activation_function = hp.Choice('activation_function', values=['relu'])
        hidden_layers = hp.Int('hidden_layers', min_value=1, max_value=3, step=1)

        model = keras.Sequential()
        inputs = tf.keras.Input(shape=[self.amount_of_train_features,])
        model.add(inputs)
        #model.add(self.normalizer)
        for _ in range(int(hidden_layers)):
            model.add(layers.Dense(hp_units, activation=hp_activation_function))
        model.add(layers.Dense(2))

        hp_learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling="log")
        model.compile(loss='mean_squared_error',
                    optimizer=tf.keras.optimizers.Adam(hp_learning_rate))
        return model


def build_and_compile_model(norm: tf.keras.layers.experimental.preprocessing.Normalization, hyper_parameters: List, amount_of_train_features: int) -> tf.keras.Sequential:
    """This function gives the NN model"""
    model = keras.Sequential()
    inputs = tf.keras.Input(shape=[amount_of_train_features,])
    model.add(inputs)
    #model.add(norm)
    for _ in range(int(hyper_parameters[3])):
        model.add(layers.Dense(hyper_parameters[0], activation = hyper_parameters[1]))
    model.add(layers.Dense(2))

    model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(float(hyper_parameters[2])))
    return model

@dataclass
class General_Simulation_parameters:
    function:str =False

def standalone_simulate_function(datapoint: DSL_Data_Point, General_Simulation_parameters, Simulation_parameters) -> DSL_Data_Point:
    """This function gives the next state for a state and control variables"""
    current_X = vertcat(datapoint.input_dataframe['theta1'].iloc[0], datapoint.input_dataframe['x1'].iloc[0], datapoint.input_dataframe['y1'].iloc[0], datapoint.input_dataframe['theta0'].iloc[0])
    current_U = vertcat(datapoint.output_dataframe['delta0'].iloc[0], datapoint.output_dataframe['v0'].iloc[0])

    system_result = General_Simulation_parameters.function(x0=current_X, u=current_U, T=0.5)["xf"]
    system_result=casadi_helpers.DM2numpy(system_result, [2,1])

    index = datapoint.input_dataframe['index'].iloc[0]
    output_datapoint = DSL_Data_Point(input = {'index': index+1, 'x1': system_result[1], 'y1': system_result[2], 'theta0': system_result[3], 'theta1': system_result[0]})

    return output_datapoint

def standalone_simulate_function_nudge(datapoint: DSL_Data_Point, General_Simulation_parameters, Simulation_parameters) -> DSL_Data_Point:
    """This function gives the next state for a state and control variables"""
    current_X = vertcat(datapoint.input_dataframe['theta1'].iloc[0], datapoint.input_dataframe['x1'].iloc[0], datapoint.input_dataframe['y1'].iloc[0], datapoint.input_dataframe['theta0'].iloc[0])
    current_U = vertcat(datapoint.output_dataframe['delta0'].iloc[0], datapoint.output_dataframe['v0'].iloc[0])

    system_result = General_Simulation_parameters.function(x0=current_X, u=current_U, T=0.5)["xf"]
    system_result=casadi_helpers.DM2numpy(system_result, [2,1])

    nudge_x1 = random.uniform(-0.01,0.01)
    nudge_y1 = random.uniform(-0.01,0.01)
    nudge_theta0 = random.uniform(-0.01,0.01)
    nudge_theta1 = random.uniform(-0.01,0.01)

    index = datapoint.input_dataframe['index'].iloc[0]
    output_datapoint = DSL_Data_Point(input = {'index': index+1, 'x1': system_result[1]+nudge_x1, 'y1': system_result[2]+nudge_y1, 'theta0': system_result[3]+nudge_theta0, 'theta1': system_result[0]+nudge_theta1})

    return output_datapoint

def initialize_ocp(theta1_init: float, x1_init: float, y1_init: float, theta0_init: float) -> List[Function]:
    """Function that initialize the ocp for the cart pole"""
    
    return np.nan

def get_perfect_paths() -> List[Data_set]:
    """This function gives the perfect paths"""
   
    return np.nan

#-------------------------------------------#
# Functions to disable and restore printing #
#-------------------------------------------#

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    # input=tf.constant([-100.0, -50.0, -5.0, -1.0, 0.0, 1.0, 5.0, 50.0, 100.0])
    # print(restricted_output(input))
    dataset = DSL_Data_Set()