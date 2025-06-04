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
from simulator import simulator_delta_init

class Truck_Trailer_Multi_Stage_Use_case(Use_Case):
    def set_self_parameters(self):
        """This function will set the self parameters"""
        self.labels_input = ['x1','y1','theta0','theta1']
        self.NN_labels_input = ['x1','y1','theta0','theta1']
        self.labels_output = ['v0','delta0','v1']
        self.hyperparameters = ['units', 'activation_function', 'learning_rate', 'hidden_layers']
        self.perfect_paths = get_perfect_paths()
        self.end_iterations = [35]*0
        self.timesteps = [1]*0
        self.timestep = 1
        #self.custom_objects = {'restricted_output_mse': restricted_output_mse}

        #[solve, Sim_cart_pole_dyn] = initialize_ocp(0, 0, 0, 0)
        # simu = simulator_delta_init()
        # self.MPC_function = solve
        # self.simulation_function = simu
        # self.expert_policy = self.expert_output_function
        # self.simulate_system = Simulate_system(self.simulate_function)
        
    def give_hypermodel(self, train_features: pd.DataFrame) -> kt.HyperModel:
        """This function gives the hyper model"""
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        hyper_model = Truck_Trailer_Multi_Stage_HyperModel(normalizer, len(self.labels_input))
        return hyper_model

    def give_NN_model(self, hyper_parameters: List, train_features: pd.DataFrame) -> tf.keras.Sequential:
        """This function gives the NN model"""
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        model = build_and_compile_model(normalizer, hyper_parameters, len(self.labels_input))
        return model
    
    def expert_output_function(self, input_datapoint: DSL_Data_Point) -> DSL_Data_Point:

        return np.nan 

    def simulate_function(self, datapoint: DSL_Data_Point, Simulation_parameters) -> DSL_Data_Point:
        """This function gives the next state for a state and control variables"""

        return np.nan 
    def give_next_state(self, datapoint: List[float], control_variables: List[float], time_step: float, idx: int) -> List[float]:
        """This function gives the next state for a state and control variables"""

        return np.nan 
    
class Truck_Trailer_Multi_Stage_HyperModel(kt.HyperModel):
    """This class contains the hypermodel"""
    def __init__(self, norm: tf.keras.layers.experimental.preprocessing.Normalization, amount_of_train_features: int):
        self.normalizer = norm
        self.amount_of_train_features = amount_of_train_features

    def build(self, hp: kt.engine.hyperparameters.HyperParameter) -> tf.keras.Sequential:
        hp_units = hp.Int('units', min_value=16, max_value=96, step=16)
        hp_activation_function = hp.Choice('activation_function', values=['sigmoid', 'tanh'])
        hidden_layers = hp.Int('hidden_layers', min_value=3, max_value=9, step=1)

        model = keras.Sequential()
        inputs = tf.keras.Input(shape=[self.amount_of_train_features,])
        model.add(inputs)
        #model.add(self.normalizer)
        for _ in range(int(hidden_layers)):
            model.add(layers.Dense(hp_units, activation=hp_activation_function))
        model.add(layers.Dense(3))

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
    model.add(layers.Dense(3))

    model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(float(hyper_parameters[2])))
    return model

@dataclass
class General_Simulation_parameters:
    function:str =False

def standalone_simulate_function(datapoint: DSL_Data_Point, General_Simulation_parameters, Simulation_parameters) -> DSL_Data_Point:
    """This function gives the next state for a state and control variables"""

    return np.nan 

def initialize_ocp(theta1_init: float, x1_init: float, y1_init: float, theta0_init: float) -> List[Function]:
    """Function that initialize the ocp for the cart pole"""
    
    return [np.nan, np.nan]

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