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

class Cart_pole_Use_case(Use_Case):
    def set_self_parameters(self):
        """This function will set the self parameters"""
        self.labels_input = ['POS','THETA','DPOS','DTHETA']
        self.NN_labels_input = ['POS','THETA','DPOS','DTHETA']
        self.labels_output = ['F']
        self.hyperparameters = ['units', 'activation_function', 'learning_rate', 'hidden_layers']
        self.perfect_paths = get_perfect_paths()
        self.end_iterations = [100]*len(self.perfect_paths)
        self.timesteps = [0.04]*len(self.perfect_paths)
        self.timestep = 0.04
        self.custom_objects = {'restricted_output_mse': restricted_output_mse}

        [solve, Sim_cart_pole_dyn] = initialize_ocp(0, 0, 0, 0)
        self.MPC_function = solve
        self.simulation_function = Sim_cart_pole_dyn
        self.expert_policy = self.expert_output_function
        self.simulate_system = Simulate_system(self.simulate_function)
        
    def give_hypermodel(self, train_features: pd.DataFrame) -> kt.HyperModel:
        """This function gives the hyper model"""
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        hyper_model = Cart_pole_HyperModel(normalizer)
        return hyper_model

    def give_NN_model(self, hyper_parameters: List, train_features: pd.DataFrame) -> tf.keras.Sequential:
        """This function gives the NN model"""
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        model = build_and_compile_model(normalizer, hyper_parameters)
        return model
    
    def expert_output_function(self, input_datapoint: DSL_Data_Point) -> DSL_Data_Point:
        MPC_input = vertcat(input_datapoint.input_dataframe['POS'].iloc[0], input_datapoint.input_dataframe['THETA'].iloc[0], input_datapoint.input_dataframe['DPOS'].iloc[0], input_datapoint.input_dataframe['DTHETA'].iloc[0])
        try:
            F = self.MPC_function(MPC_input)
            F = casadi_helpers.DM2numpy(F, [2,1])
            output_datapoint = DSL_Data_Point(output = {'F' : F})
        except:
            output_datapoint = DSL_Data_Point()

        return output_datapoint
        
    # def give_expert_actions(self, input: List[List[float]]) -> List[List[float]]:
    #     """This function gives expert actions for a list of states"""
    #     output_F = []

    #     for idx in range(len(input[0])):
    #         [F] = self.give_expert_action([input[0][idx], input[1][idx], input[2][idx], input[3][idx]]) 
    #         output_F.append(F)

    #     return [output_F]
    
    # def give_expert_action(self, input: List[float], iterations=0) -> List[float]:
    #     """This function gives an expert action for a state"""
    #     try:
    #         F = self.MPC_function(vertcat( input[0], input[1], input[2], input[3]))
    #         F = casadi_helpers.DM2numpy(F, [2,1])
    #         return [F]
    #     except:
    #         return [False]  

    def simulate_function(self, datapoint: DSL_Data_Point, Simulation_parameters) -> DSL_Data_Point:
        """This function gives the next state for a state and control variables"""
        current_X = vertcat(datapoint.input_dataframe['POS'].iloc[0], datapoint.input_dataframe['THETA'].iloc[0], datapoint.input_dataframe['DPOS'].iloc[0], datapoint.input_dataframe['DTHETA'].iloc[0])
        current_U = vertcat(datapoint.output_dataframe['F'].iloc[0])

        system_result = self.simulation_function(x0=current_X, u=current_U, T=self.time_step)["xf"]
        system_result=casadi_helpers.DM2numpy(system_result, [2,1])

        output_datapoint = DSL_Data_Point(input = {'POS': system_result[0], 'THETA': system_result[1],'DPOS': system_result[2],'DTHETA': system_result[3]})

        return output_datapoint
    def give_next_state(self, datapoint: List[float], control_variables: List[float], time_step: float, idx: int) -> List[float]:
        """This function gives the next state for a state and control variables"""
        current_X = vertcat(input[0], input[1], input[2], input[3])
        current_U = vertcat(control_variables[0])

        system_result = self.simulation_function(x0=current_X, u=current_U, T=time_step)["xf"]
        system_result=casadi_helpers.DM2numpy(system_result, [2,1])

        return [system_result[0], system_result[1], system_result[2], system_result[3]]

    def sample_k_start_points(self, k: int):
        """This function gives k start points"""
        start_points = []

        for _ in range(k):
            start_pos = uniform(-0.5, 0.5)

            start_point = [start_pos, 0, 0, 0]
            start_points.append(start_point)
        
        end_iterations = [100]*k
        
        return start_points, end_iterations

class Cart_pole_HyperModel(kt.HyperModel):
    """This class contains the hypermodel"""
    def __init__(self, norm: tf.keras.layers.experimental.preprocessing.Normalization):
        self.normalizer = norm

    def build(self, hp: kt.engine.hyperparameters.HyperParameter) -> tf.keras.Sequential:
        hp_units = hp.Int('units', min_value=16, max_value=96, step=16)
        hp_activation_function = hp.Choice('activation_function', values=['relu'])
        hidden_layers = hp.Int('hidden_layers', min_value=4, max_value=4, step=1)

        model = keras.Sequential()
        model.add(self.normalizer)
        for _ in range(int(hidden_layers)):
            model.add(layers.Dense(hp_units, activation=hp_activation_function))
        model.add(layers.Dense(1))

        hp_learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling="log")
        model.compile(loss='mean_squared_error',
                    optimizer=tf.keras.optimizers.Adam(hp_learning_rate))
        return model

def build_and_compile_model(normalizer, hyper_parameters: List) -> tf.keras.Sequential:
    """This function gives the NN model"""
    # normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
    # normalizer.adapt(np.array(train_features))
        
    model = keras.Sequential()
    model.add(normalizer)
    for _ in range(int(hyper_parameters[3])):
        model.add(layers.Dense(hyper_parameters[0], activation = hyper_parameters[1]))
    model.add(layers.Dense(1))

    model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(hyper_parameters[2]))
    return model

@dataclass
class General_Simulation_parameters:
    function:str =False

def standalone_simulate_function(datapoint: DSL_Data_Point, General_Simulation_parameters, Simulation_parameters) -> DSL_Data_Point:
    """This function gives the next state for a state and control variables"""
    current_X = vertcat(datapoint.input_dataframe['POS'].iloc[0], datapoint.input_dataframe['THETA'].iloc[0], datapoint.input_dataframe['DPOS'].iloc[0], datapoint.input_dataframe['DTHETA'].iloc[0])
    current_U = vertcat(datapoint.output_dataframe['F'].iloc[0])

    system_result = General_Simulation_parameters.function(x0=current_X, u=current_U, T=0.04)["xf"]
    system_result=casadi_helpers.DM2numpy(system_result, [2,1])

    output_datapoint = DSL_Data_Point(input = {'POS': system_result[0], 'THETA': system_result[1],'DPOS': system_result[2],'DTHETA': system_result[3]})

    return output_datapoint

def initialize_ocp(pos_init: float, theta_init: float, dpos_init: float, dtheta_init: float) -> List[Function]:
    """Function that initialize the ocp for the cart pole"""
    # -------------------------------
    # Problem parameters
    # -------------------------------
    mcart = 0.5                 # cart mass [kg]
    m     = 1                   # pendulum mass [kg]
    L     = 2                   # pendulum length [m]
    g     = 9.81                # gravitation [m/s^2]

    nx    = 4                   # the system is composed of 4 states
    nu    = 1                   # the system has 1 input
    Tf    = 2.0                 # control horizon [s]
    Nhor  = 50                  # number of control intervals
    dt    = Tf/Nhor             # sample time
   
    current_X = vertcat(pos_init,theta_init,dpos_init,dtheta_init)  # initial state
    final_X   = vertcat(0,0,0,0)    # desired terminal state

    # -------------------------------
    # Set OCP
    # -------------------------------
    ocp = Ocp(T=Tf)

    # Define states
    pos    = ocp.state()  # [m]
    theta  = ocp.state()  # [rad]
    dpos   = ocp.state()  # [m/s]
    dtheta = ocp.state()  # [rad/s]

    # Defince controls
    F = ocp.control(nu, order=0)

    # Define parameter
    X_0 = ocp.parameter(nx);

    # Specify ODE
    ocp.set_der(pos, dpos)
    ocp.set_der(theta, dtheta)
    ocp.set_der(dpos, (-m*L*sin(theta)*dtheta*dtheta + m*g*cos(theta)*sin(theta)+F)/(mcart + m - m*cos(theta)*cos(theta)) )
    ocp.set_der(dtheta, (-m*L*cos(theta)*sin(theta)*dtheta*dtheta + F*cos(theta)+(mcart+m)*g*sin(theta))/(L*(mcart + m - m*cos(theta)*cos(theta))))

    # Lagrange objective
    ocp.add_objective(ocp.integral(F*2 + 100*pos**2))

    # Path constraints
    ocp.subject_to(-2 <= (F <= 2 ))
    ocp.subject_to(-2 <= (pos <= 2))

    # Initial constraints
    X = vertcat(pos,theta,dpos,dtheta)
    ocp.subject_to(ocp.at_t0(X)==X_0)
    ocp.subject_to(ocp.at_tf(X)==final_X)

    # Pick a solution method
    options = {"ipopt": {"print_level": 0}}
    options["expand"] = True
    options["print_time"] = False
    ocp.solver('ipopt',options)

    # Make it concrete for this ocp
    ocp.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

    ocp.set_value(X_0, current_X)
    solve = ocp.to_function('solve',
    [X_0],
    [ocp.sample(F,grid='control')[1][0]]) 

    Sim_cart_pole_dyn = ocp._method.discrete_system(ocp)
   
    return [solve, Sim_cart_pole_dyn]

def get_perfect_paths() -> List[Data_set]:
    """This function gives the perfect paths"""

    with open(str(os.path.join("MPC_data", "MPC_position_mpc_example.csv"))) as file_name:
    # with open("MPC_data\MPC_position_mpc_example.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_pos = []
        for row in csvreader:
            MPC_pos.append([float(i) for i in row])
        
    with open(str(os.path.join("MPC_data", "MPC_theta_mpc_example.csv"))) as file_name:
    # with open("MPC_data\MPC_theta_mpc_example.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_theta = []
        for row in csvreader:
            MPC_theta.append([float(i) for i in row])

    with open(str(os.path.join("MPC_data", "MPC_dposition_mpc_example.csv"))) as file_name:
    # with open("MPC_data\MPC_dposition_mpc_example.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_dpos = []
        for row in csvreader:
            MPC_dpos.append([float(i) for i in row])

    with open(str(os.path.join("MPC_data", "MPC_dtheta_mpc_example.csv"))) as file_name:
    # with open("MPC_data\MPC_dtheta_mpc_example.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_dtheta = []
        for row in csvreader:
            MPC_dtheta.append([float(i) for i in row])
    
    perfect_paths = []
    
    for idx in range(len(MPC_pos)):
        perfect_path = Data_set(4, 0)
        perfect_path.load_data_from_col_list([MPC_pos[idx], MPC_theta[idx], MPC_dpos[idx], MPC_dtheta[idx]])
        perfect_paths.append(perfect_path)
   
    return perfect_paths

def restricted_output(x):
    """This function is the restricted output activation function"""
    min_edge = -2.0
    max_edge = 2.0
    return tf.math.sigmoid(x)*(max_edge-min_edge)+min_edge

def restricted_output_mse(y_true, y_pred):
    """This function is the restricted output loss function"""
    big_value = 1e5

    min_edge = -2.0
    max_edge = 2.0

    y_shape = K.shape(y_pred).shape
 
    min_edge = K.constant(min_edge, shape = y_shape)
    max_edge = K.constant(max_edge, shape = y_shape)

    switch_cond_max_edge = K.cast(K.greater_equal(max_edge, y_pred), "int32")
    switch_cond_min_edge = K.cast(K.greater_equal(y_pred, min_edge), "int32")
    switch_cond = K.minimum(switch_cond_max_edge, switch_cond_min_edge)

    loss_tensor = K.switch(switch_cond, K.square(y_true-y_pred), K.square(y_true-y_pred)*big_value)
    
    loss = K.mean(loss_tensor)
    return loss
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
