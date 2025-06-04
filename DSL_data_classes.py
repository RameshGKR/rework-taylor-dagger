import warnings
import pandas as pd
import numpy as np
import math
import os
from pathlib import Path
from io import StringIO
from os.path import exists
from typing import Dict, List, Any

class DSL_Data_Point:
    def __init__(self, input:Dict = {}, output:Dict = {}):
        """
        Initialize the datapoint using dictionaries
        
        Args:
            - input (Dict): data to initialize input side of datapoint (optional)
            - output (Dict): data to initialize output side of datapoint (optional)
        """
        self.input_dataframe = pd.DataFrame()
        self.output_dataframe = pd.DataFrame()

        self.input_labels = []
        self.output_labels = []

        self.error_included = False

        self.initialize_with_dictionaries(input, output)

    def copy(self):
        """
        returns a copy of the datapoint
        """
        copy_data_point = DSL_Data_Point()
        copy_data_point.input_dataframe = self.input_dataframe
        copy_data_point.output_dataframe = self.output_dataframe
        copy_data_point.input_labels = self.input_labels
        copy_data_point.output_labels = self.output_labels
        return copy_data_point

    def __eq__(self, input_datapoint):
        """
        checks if input datapoint is equal to the datapoint
        
        Args:
            - input datapoint that has to be checked
        """
        if not isinstance(input_datapoint, DSL_Data_Point):
            raise ValueError("Can't compare with an object that is not a Dataset")
        
        bool_input_df = self.input_dataframe.equals(input_datapoint.input_dataframe)
        bool_output_df = self.output_dataframe.equals(input_datapoint.output_dataframe)
        bool_input_labels = self.input_labels == input_datapoint.input_labels
        bool_output_labels = self.output_labels == input_datapoint.output_labels

        total_bool = bool_input_df and bool_output_df and bool_input_labels and bool_output_labels
        return total_bool

    def __setattr__(self, __name: str, __value: Any) -> None:
        """
        function that allows to change certain variables of the datapoint two extra options are here created "input" and "output".
        Input gives the possibility to change the input side of the datapoint using a datapoint with only input parameters.
        Output gives the possibility to change the output side of the datapoint using a datapoint with only output parameters.
        """

        if __name == "input":
            if not isinstance(__value, DSL_Data_Point):
                raise AttributeError("input has to be a datapoint")
            if not __value.output_dataframe.empty:
                raise AttributeError("the output of the datapoint has to be empty")
            if __value.input_dataframe.empty:
                warnings.warn("The input of the datapoint is empty")
            
            self.initialize_with_dataframe(input=__value.input_dataframe)
        elif __name == "output":
            if not isinstance(__value, DSL_Data_Point):
                raise AttributeError("output has to be a data point")
            if not __value.input_dataframe.empty:
                raise AttributeError("the input of the datapoint has to be empty")
            if __value.output_dataframe.empty:
                warnings.warn("The output of the datapoint is empty")
            
            self.initialize_with_dataframe(output=__value.output_dataframe)
        else:
            super().__setattr__(__name, __value)

    def __getattribute__(self, __name: str) -> Any:
        """
        function that allows to get certain variables of the datapoint two extra options are here created "input" and "output".
        Input gives the possibility to get only the input side of the datapoint as a datapoint.
        Output gives the possibility to get only the output side of the datapoint as a datapoint.
        """
        if __name == "input":
            input_datapoint = DSL_Data_Point()
            input_datapoint.initialize_with_dataframe(input=self.input_dataframe)
            return input_datapoint
        elif __name == "output":
            output_datapoint = DSL_Data_Point()
            output_datapoint.initialize_with_dataframe(output=self.output_dataframe)
            return output_datapoint
        else:
            return super(DSL_Data_Point, self).__getattribute__(__name)

    def initialize_with_dictionaries(self, input:Dict = {}, output:Dict = {}):
        """
        Allows to initialize a datapoint using dictionaries
        
        Args:
            - dictionary to initialize input side of datapoint
            - dictionary to initialize output side of datapoint
        """
        if not isinstance(input, Dict):
            raise ValueError("The incoming input data has to be a dictionary")
        if not isinstance(output, Dict):
            raise ValueError("The incoming output data has to be a dictionary")
        
        self.input_dataframe = pd.DataFrame(input, index=[0])
        self.output_dataframe = pd.DataFrame(output, index=[0])

        self.input_labels = list(input.keys())
        self.output_labels = list(output.keys())

    def initialize_with_dataframe(self, input=pd.DataFrame(), output=pd.DataFrame()):
        """
        Allows to initialize a datapoint using dataframes
        
        Args:
            - input (pd.DataFrame): data to initialize input side of datapoint
            - output (pd.DataFrame): data to initialize output side of datapoint
        """
        if not isinstance(input, pd.DataFrame):
            raise ValueError("The incoming input data has to be a dataframe")
        if not input.shape[0] == 1 and not input.shape[0] == 0:
            raise ValueError("The incoming input data has to be empty or have a length of 1")

        if input.shape[0] == 1:
            self.input_dataframe = input.set_index(pd.Index([0]))
            self.input_labels =list(input.keys())

        if not isinstance(output, pd.DataFrame):
            raise ValueError("The incoming output data has to be a dataframe")
        if not output.shape[0] == 1 and not output.shape[0] == 0:
            raise ValueError("The incoming output data has to be empty or have a length of 1")

        if output.shape[0] == 1:
            self.output_dataframe = output.set_index(pd.Index([0]))
            self.output_labels =list(output.keys())
            
class DSL_Data_Set:
    def __init__(self, input_dataframe=pd.DataFrame(), output_dataframe=pd.DataFrame(), is_trace:bool = False):
        """
        initialize a dataset the only parameter that is set is the is_trace parameter
        
        Args:
            - input (pd.DataFrame): data to initialize input side of datapoint (optional)
            - output (pd.DataFrame): data to initialize output side of datapoint (optional)
            - is_trace (bool): Bool that signifies if dataset is a trace or not
        """
        self.input_dataframe= pd.DataFrame()
        self.output_dataframe= pd.DataFrame()
        self.input_labels = []
        self.output_labels = []
        self.length = 0
        self.is_trace = is_trace
        self.error_included = False
        self.random_seed = 37

        self.initialize_with_dataframe(input_dataframe, output_dataframe)
    
    def copy(self):
        """
        returns a copy of the dataset
        """
        copy_data_set = DSL_Data_Set()
        copy_data_set.input_dataframe = self.input_dataframe
        copy_data_set.output_dataframe = self.output_dataframe
        copy_data_set.input_labels = self.input_labels
        copy_data_set.output_labels = self.output_labels
        copy_data_set.length = self.length
        copy_data_set.is_trace = self.is_trace
        copy_data_set.random_seed = self.random_seed
        return copy_data_set

    def __setattr__(self, __name: str, __value: Any) -> None:
        """
        function that allows to change certain variables of the dataset two extra options are here created "input" and "output".
        Input gives the possibility to change the input side of the dataset using a dataset with only input parameters.
        Output gives the possibility to change the output side of the dataset using a dataset with only output parameters.
        """
        if __name == "input":
            if not isinstance(__value, DSL_Data_Set):
                raise AttributeError("input has to be a dataset")
            if not __value.output_dataframe.empty:
                raise AttributeError("the output of the dataset has to be empty")
            if not self.output_dataframe.empty and self.output_dataframe.shape[0] != __value.input_dataframe.shape[0]:
                raise AttributeError("the input of the dataset has to have the same length as the existing output in the dataset if the output is not empty")
            if __value.input_dataframe.empty:
                warnings.warn("The input of the dataset is empty")
                return
            
            self.input_dataframe= __value.input_dataframe
            self.input_labels = list(__value.input_dataframe.keys())
            self.length = __value.input_dataframe.shape[0]
        elif __name == "output":
            if not isinstance(__value, DSL_Data_Set):
                raise AttributeError("output has to be a dataset")
            if not __value.input_dataframe.empty:
                raise AttributeError("the input of the dataset has to be empty")
            if not self.input_dataframe.empty and self.input_dataframe.shape[0] != __value.output_dataframe.shape[0]:
                raise AttributeError("the output of the dataset has to have the same length as the existing input in the dataset if the input is not empty")
            if __value.output_dataframe.empty:
                warnings.warn("The output of the dataset is empty")
                return
            
            self.output_dataframe=__value.output_dataframe
            self.output_labels = list(__value.output_dataframe.keys())
            self.length = __value.output_dataframe.shape[0]     
        else:
            super().__setattr__(__name, __value)

    def __getattribute__(self, __name: str) -> Any:
        """
        function that allows to get certain variables of the datapoint two extra options are here created "input" and "output".
        Input gives the possibility to get only the input side of the datapoint as a datapoint.
        Output gives the possibility to get only the output side of the datapoint as a datapoint.
        """
        if __name == "input":
            input_dataset = DSL_Data_Set()
            input_dataset.initialize_with_dataframe(input_dataframe=self.input_dataframe)
            return input_dataset
        elif __name == "output":
            output_dataset = DSL_Data_Set()
            output_dataset.initialize_with_dataframe(output_dataframe=self.output_dataframe)
            return output_dataset
        else:
            return super(DSL_Data_Set, self).__getattribute__(__name)

    def __iter__(self):
        """
        function to iterate over a dataset
        """
        self.index = 0
        return self
 
    def __next__(self):
        """
        function to iterate over a dataset
        """
        index = self.index
 
        if index >= self.length:
            raise StopIteration
 
        self.index = index + 1

        datapoint = self.get_iter_datapoint_from_dataset(index)

        return datapoint

    def __eq__(self, other):
        """
        checks if two datasets are equal
        """
        if not isinstance(other, DSL_Data_Set):
            raise ValueError("Can't compare with an object that is not a Dataset")
        
        bool_input_df = self.input_dataframe.equals(other.input_dataframe)
        bool_output_df = self.output_dataframe.equals(other.output_dataframe)
        bool_input_labels = self.input_labels == other.input_labels
        bool_output_labels = self.output_labels == other.output_labels
        bool_length = self.length == other.length
        bool_is_trace = self.is_trace == other.is_trace
        bool_random_seed = self.random_seed == other.random_seed

        total_bool = bool_input_df and bool_output_df and bool_input_labels and bool_output_labels and bool_length and bool_is_trace and bool_random_seed

        return total_bool

    def get_iter_datapoint_from_dataset(self, index: int):
        """
        returns certain datapoint according to the index of the datapoint

        Args:
            - index (int): index of the datapoint that is extracted from the dataset
        """
        datapoint = DSL_Data_Point()

        if not self.input_dataframe.empty:
            datapoint_input = self.input_dataframe.iloc[[index]]
            datapoint_input = datapoint_input.set_index(pd.Index([0]))
            datapoint.initialize_with_dataframe(input=datapoint_input)
        if not self.output_dataframe.empty:
            datapoint_output = self.output_dataframe.iloc[[index]]
            datapoint_output = datapoint_output.set_index(pd.Index([0]))
            datapoint.initialize_with_dataframe(output=datapoint_output)
        
        return datapoint

    def initialize_with_dataframe(self, input_dataframe=pd.DataFrame(), output_dataframe=pd.DataFrame()):
        """
        Allows to initialize a datapoint using dataframes
                
        Args:
            - input_dataframe (pd.DataFrame): data to initialize input side of datapoint
            - output_dataframe (pd.DataFrame): data to initialize output side of datapoint
        """
        if not isinstance(input_dataframe, pd.DataFrame):
            raise ValueError("The incoming input data has to be a dataframe")
        if not isinstance(output_dataframe, pd.DataFrame):
            raise ValueError("The incoming output data has to be a dataframe")
        if not input_dataframe.empty and not output_dataframe.empty and (input_dataframe.shape[0] != output_dataframe.shape[0]):
            raise ValueError("The incoming input data and output data are not empty and are not the same length")
        # if input.empty and output.empty:
        #     warnings.warn("The input and output dataframes are empty")
        #     return

        if not input_dataframe.empty:
            self.input_dataframe = input_dataframe
            self.input_labels = list(input_dataframe.keys())
            self.length = input_dataframe.shape[0]
        if not output_dataframe.empty:
            self.output_dataframe = output_dataframe
            self.output_labels = list(output_dataframe.keys())
            self.length = output_dataframe.shape[0]

    def initialize_from_csv(self, file_name):
        """Allows to initialize a dataset using a csv file"""
        try:
            dataframe = pd.read_csv(file_name)
        except:
            raise ValueError("The csv file could not be read")
        
        if Path(file_name).suffix != ".csv":
            raise ValueError("The file is not a csv file")
        
        if dataframe.empty:
            warnings.warn("The csv file is empty")
            return

        input_labels, output_labels = self.__get_labels_from_csv_dataframe(dataframe)
        input_dataframe, output_dataframe = self.__split_and_label_csv_dataframe(dataframe, input_labels, output_labels)

        if not input_dataframe.empty:
            self.input_dataframe= input_dataframe
            self.input_labels = input_labels
            self.length = input_dataframe.shape[0]
        if not output_dataframe.empty:
            self.output_dataframe= output_dataframe
            self.output_labels = output_labels
            self.length = output_dataframe.shape[0]
 
    def __get_labels_from_csv_dataframe(self, dataframe):
        """returns the input and output labels from the dataframe gotten from the csv file"""
        labels = list(dataframe.keys())
        input_labels = []
        output_labels = []

        for label in labels:
            if label.startswith('input_'):
                input_labels.append(label.split('input_',1)[1])
            elif label.startswith(' input_'):
                input_labels.append(label.split(' input_',1)[1])
            elif label.startswith('output_'):
                output_labels.append(label.split('output_',1)[1])
            elif label.startswith(' output_'):
                output_labels.append(label.split(' output_',1)[1])
            else:
                raise ValueError("The column names in the csv file have to start with 'input_' or 'output_'")
        
        return input_labels, output_labels

    def __split_and_label_csv_dataframe(self, dataframe, input_labels, output_labels):
        """returns the input and output dataframes from the dataframe gotten from the csv file"""
        input_dataframe = dataframe.iloc[:,:len(input_labels)]
        output_dataframe = dataframe.iloc[:,len(input_labels):]

        for idx, label in enumerate(input_labels):
            input_dataframe.columns.values[idx]=label

        for idx, label in enumerate(output_labels):
            output_dataframe.columns.values[idx]=label

        return input_dataframe, output_dataframe

    def append_datapoint(self, datapoint):
        """allows to append a datapoint to the dataset"""
        if not isinstance(datapoint, DSL_Data_Point):
            raise ValueError("The incoming data has to be a Data_Point")
        if not any(datapoint.input_dataframe) and not any(datapoint.output_dataframe):
            warnings.warn("The datapoint is empty")
            return

        if datapoint.error_included:
            self.error_included = True

        self.__check_for_label_errors(datapoint)
        self.__check_for_append_errors(datapoint)

        self.__set_labels(datapoint)
        self.__append_data(datapoint)
        self.length = self.length + 1
       
    def append_dataset(self, dataset):
        """allows to append a dataset to the dataset"""
        if not isinstance(dataset, DSL_Data_Set):
            raise ValueError("The incoming data has to be a Data_Set")
        if not dataset.length>0:
            warnings.warn("The dataset is empty")
            return

        if dataset.error_included:
            self.error_included = True

        self.__check_for_label_errors(dataset)
        self.__check_for_append_errors(dataset)

        self.__set_labels(dataset)
        self.__append_data(dataset)
        self.length = self.length + dataset.length
      
    def append_trace_dataset(self, trace_dataset):
        """allows to append a trace dataset to a dataset"""
        if not isinstance(trace_dataset, DSL_Trace_Data_Set):
            raise ValueError("The incoming data has to be a Trace_Data_Set")
        if not any(trace_dataset.datasets):
            warnings.warn("The trace dataset is empty")
            return

        self.__check_for_label_errors(trace_dataset)
        self.__check_for_append_errors(trace_dataset.datasets[0])

        self.__set_labels(trace_dataset)
        for dataset in trace_dataset.datasets:
            self.__append_data(dataset)
            self.length = self.length + dataset.length

    def remove_datapoints(self, indexes):
        """remove certain datapoint given the index of these datapoints"""
        if self.is_trace:
            raise ValueError("datapoints can't be removed from a trace")
        if not isinstance(indexes, list):
            raise ValueError("The indexes has to be a list")
        indexes=list(set(indexes)) #get all the unique indexes in the list
        for index in indexes:
            if not isinstance(index, int):
                raise ValueError("The index has to be a integer")
            if (0>index) and (self.length<=index):
                raise ValueError("The index has to between 0 and the length of the dataset")
        
        self.input_dataframe=self.input_dataframe.drop(indexes)
        self.input_dataframe=self.input_dataframe.reset_index(drop=True)
        self.output_dataframe=self.output_dataframe.drop(indexes)
        self.output_dataframe=self.output_dataframe.reset_index(drop=True)
        self.length = self.length-len(indexes)

    def __check_for_append_errors(self, incoming_data):
        """checks if the incoming data can be appended to the existing data in the dataset"""
        sum_bools = int(any(self.input_dataframe)) + int(any(self.output_dataframe)) + int(any(incoming_data.input_dataframe)) + int(any(incoming_data.output_dataframe))

        if sum_bools == 3:
            raise ValueError("The incoming data can't be appended to the existing data")
        
        if any(incoming_data.input_dataframe) and not any(self.input_dataframe) and any(self.output_dataframe):
            raise ValueError("The incoming data can't be appended to the existing data")

        if any(incoming_data.output_dataframe) and not any(self.output_dataframe) and any(self.input_dataframe):
            raise ValueError("The incoming data can't be appended to the existing data")

    def __check_for_label_errors(self, incoming_data):
        """check if the labels of the incoming data match up with the existing labels in the dataset"""
        if self.input_labels and incoming_data.input_labels and (self.input_labels != incoming_data.input_labels):
            raise ValueError("The incoming input labels and the existing input labels don't match")

        if self.output_labels and incoming_data.output_labels and (self.output_labels != incoming_data.output_labels):
            raise ValueError("The incoming output labels and the existing output labels don't match")

    def __append_data(self, incoming_data):
        """append the data to the dataset"""
        if any(incoming_data.input_dataframe):
            self.input_dataframe= pd.concat([self.input_dataframe, incoming_data.input_dataframe])
            self.input_dataframe = self.input_dataframe.reset_index(drop=True)
        
        if any(incoming_data.output_dataframe):
            self.output_dataframe= pd.concat([self.output_dataframe, incoming_data.output_dataframe])
            self.output_dataframe = self.output_dataframe.reset_index(drop=True)

    def __set_labels(self, incoming_data):
        """if there are no existing labels this function will set the labels"""
        if not self.input_labels and incoming_data.input_labels:
            self.input_labels = incoming_data.input_labels
        
        if not self.output_labels and incoming_data.output_labels:
            self.output_labels = incoming_data.output_labels
    
    def shuffle(self):
        """shuffle the datapoints in the dataset"""
        if self.is_trace:
            raise ValueError("A dataset that is a trace can't be shuffled")

        if any(self.input_dataframe):
            self.input_dataframe = self.input_dataframe.sample(frac=1, random_state=self.random_seed)
            self.input_dataframe = self.input_dataframe.reset_index(drop=True)
        
        if any(self.output_dataframe):
            self.output_dataframe = self.output_dataframe.sample(frac=1, random_state=self.random_seed)
            self.output_dataframe = self.output_dataframe.reset_index(drop=True)
    
    def split_dataset(self, percentages: List[float]):
        """split the dataset into multiple dataset according to a list of percentages that signify how many of the datapoint should go into each dataset"""
        if self.is_trace:
            raise ValueError("A dataset that is a trace can't be split")
        if not (0.99 <= math.fsum(percentages) <= 1.01):
            raise ValueError("The given percentages have to add up to 1")
        if self.length == 0:
            warnings.warn("The dataset is empty")
        
        amounts = self.__find_split_amounts(percentages)
        datasets = self.__split_dataset_using_amounts(amounts)

        return datasets
        
    def remove_errors(self):
        index_list=[]
        for idx, datapoint in enumerate(self):
            if datapoint.input_dataframe.isnull().values.any() or datapoint.output_dataframe.isnull().values.any():
                index_list.append(idx)

        self.remove_datapoints(index_list)

    def __find_split_amounts(self, percentages):
        """helperfunction that finds the amount of datapoints that have to go into each dataset"""
        amounts=[]
        # get the minimum amount for all the percentages
        for percentage in percentages:
            amounts.append(np.floor(percentage*self.length))

        amounts = [int(x) for x in amounts]
        total_amount = sum(amounts)
        #if there are datapoint left add them in a way to minimize the error between the actual percentage and the asked percentage
        for _ in range(int(self.length-total_amount)):
            amounts = self.__find_best_new_amounts(amounts, percentages)
        
        return amounts

    def __find_best_new_amounts(self,amounts, percentages):
        """helperfunction that adds the remaining datapoints to the amount of datapoints in a way to minimize the error between the actual percentage and the asked percentage"""
        min_error = 100
        best_amounts = amounts

        for index in range(len(amounts)):
            current_amounts = amounts.copy()
            current_amounts[index] += 1

            error = self.__calculate_percentages_error(current_amounts, percentages)

            if error < min_error:
                min_error = error
                best_amounts = current_amounts

        return best_amounts

    def __calculate_percentages_error(self, current_amounts, percentages):
        """helperfunction that calculates the error between the actual percentage and the asked percentage"""
        current_amounts_array = np.array(current_amounts)
        percentages_array = np.array(percentages)

        current_percentages_array = current_amounts_array/self.length
        error = (np.square(current_percentages_array - percentages_array)).mean() # mean square error

        return error

    def __split_dataset_using_amounts(self, amounts):
        """helper function that shuffles and split the dataset given the amount of datapoints that have to be in each dataset"""
        self.shuffle()
        datasets = []
        previous_amount = 0

        for amount in amounts:
            dataset = DSL_Data_Set()

            if not self.input_dataframe.empty and self.output_dataframe.empty:
                new_input = self.input_dataframe.iloc[previous_amount:previous_amount+amount,:]
                new_input = new_input.reset_index(drop=True)
                dataset.initialize_with_dataframe(input_dataframe=new_input)
            elif self.input_dataframe.empty and not self.output_dataframe.empty:
                new_output = self.output_dataframe.iloc[previous_amount:previous_amount+amount,:]
                new_output = new_output.reset_index(drop=True)
                dataset.initialize_with_dataframe(output_dataframe=new_output)
            elif not self.input_dataframe.empty and not self.output_dataframe.empty:
                new_input = self.input_dataframe.iloc[previous_amount:previous_amount+amount,:]
                new_input = new_input.reset_index(drop=True)
                new_output = self.output_dataframe.iloc[previous_amount:previous_amount+amount,:]
                new_output = new_output.reset_index(drop=True)
                dataset.initialize_with_dataframe(input_dataframe=new_input, output_dataframe=new_output)
            
            datasets.append(dataset)

            previous_amount += amount
        
        return datasets
            
    def write_dataset_to_csv(self, file_name, mode='w'):
        """write the dataset to a csv file"""
        if not isinstance(file_name, str) and not os.path.isfile(file_name):
            raise ValueError("The file name should be a string and refer to a file")
        if Path(file_name).suffix != ".csv":
            raise ValueError("The file is not a csv file")

        self.input_dataframe= self.input_dataframe.reset_index(drop=True)
        self.output_dataframe= self.output_dataframe.reset_index(drop=True)
        total_dataset = pd.concat([self.input_dataframe, self.output_dataframe], axis=1)

        input_labels = ["input_"+label for label in self.input_labels]
        output_labels = ["output_"+label for label in self.output_labels]
        total_labels = input_labels+output_labels

        for idx, label in enumerate(total_labels):
            total_dataset.columns.values[idx]=label

        total_dataset.to_csv(file_name, index=False, mode=mode)
  
class DSL_Trace_Data_Set:
    def __init__(self):
        """initialize a trace dataset"""
        self.datasets = []
        self.input_labels = []
        self.output_labels = []
    
    def copy(self):
        """returns a copy of the trace dataset"""
        copy_trace_data_set = DSL_Trace_Data_Set()
        copy_trace_data_set.datasets = self.datasets
        copy_trace_data_set.input_labels = self.input_labels
        copy_trace_data_set.output_labels = self.output_labels
        return copy_trace_data_set

    def __iter__(self):
        """function to iterate over a dataset"""
        self.index = 0
        return self
 
    def __next__(self):
        """function to iterate over a dataset"""
        index = self.index
 
        if index >= len(self.datasets):
            raise StopIteration
 
        self.index = index + 1

        dataset = self.datasets[index]

        return dataset

    def __eq__(self, other):
        """checks if two trace datasets are equal"""
        if not isinstance(other, DSL_Trace_Data_Set):
            raise ValueError("Can't compare with an object that is not a Dataset")
        
        bool_datasets = self.datasets == other.datasets
        bool_input_labels = self.input_labels == other.input_labels
        bool_output_labels = self.output_labels == other.output_labels

        total_bool = bool_datasets and bool_input_labels and bool_output_labels
        return total_bool

    def append_dataset(self, dataset):
        """allows to append a dataset to the trace dataset"""
        if not isinstance(dataset, DSL_Data_Set):
            raise ValueError("The incoming data has to be a Data_Set")
        if not dataset.is_trace:
            raise ValueError("The incoming dataset is not a trace")
        if not dataset.length > 0:
            warnings.warn("The dataset is empty")
            print('help')
            return
        if any(self.datasets):
            if not self.datasets[0].length == dataset.length:
                raise ValueError("The incoming data can't be appended to the existing data")
            
        self.__check_for_label_errors(dataset)
        self.__check_for_append_errors(dataset)

        self.__set_labels(dataset)
        self.datasets.append(dataset)
        
    def append_dataset_pointwise(self, dataset):
        """allows to append a dataset pointwise to the trace dataset. Pointwise means that each datapoint in the incoming dataset goes to a seperate trace in the trace dataset"""
        if not isinstance(dataset, DSL_Data_Set):
            raise ValueError("The incoming data has to be a Data_Set")
        if dataset.is_trace:
            raise ValueError("The incoming dataset is a trace")
        if dataset.length == 0:
            warnings.warn("The incoming dataset is empty")
            return
        if not(len(self.datasets) == dataset.length or len(self.datasets) == 0):
            raise ValueError("The length of the incoming dataset has to be equal to the amount of datasets in the trace dataset if the trace dataset is not empty")
        
        self.__check_for_label_errors(dataset)
        self.__check_for_append_errors(dataset)

        self.__set_labels(dataset)

        if len(self.datasets) == 0:
            for _ in range(dataset.length):
                empty_dataset_loop = DSL_Data_Set(is_trace=True)
                self.datasets.append(empty_dataset_loop)

        for index, datapoint in enumerate(dataset):
            self.datasets[index].append_datapoint(datapoint)

    def append_trace_dataset(self, trace_dataset):
        """allows to append a trace dataset to the trace dataset"""
        if not isinstance(trace_dataset, DSL_Trace_Data_Set):
            raise ValueError("The incoming data has to be a Trace_Data_Set")
        if not any(trace_dataset.datasets):
            warnings.warn("The trace dataset is empty")
            return
        if any(self.datasets) and any(trace_dataset.datasets):
            if not self.datasets[0].length == trace_dataset.datasets[0].length:
                raise ValueError("The incoming data can't be appended to the existing data")

        self.__check_for_label_errors(trace_dataset)
        self.__check_for_append_errors(trace_dataset.datasets[0])

        self.__set_labels(trace_dataset)
        self.datasets = self.datasets + trace_dataset.datasets

    def initialize_from_csv(self, file_name):
        """Allows to initialize a dataset using a csv file"""
        if not isinstance(file_name, str) or not exists(file_name):
            raise ValueError("The csv file could not be read")
        if Path(file_name).suffix != ".csv":
            raise ValueError("The file is not a csv file")
        
        with open(file_name,'r') as input_file:
            data_str = input_file.read()
            data_array = data_str.split('\n\n\n') # Split on all instances of double new lines
            #input_output_line = data_array[0].split('\n')[0]

            for smaller_data in data_array:        
                dataset=self.__initialize_from_csv_for_dataset__(smaller_data)
                self.append_dataset(dataset)

    def __initialize_from_csv_for_dataset__(self, data):
        dataframe = pd.read_csv(StringIO(data))
        input_labels, output_labels = self.__get_labels_from_csv_dataframe(dataframe)
        input_dataframe, output_dataframe = self.__split_and_label_csv_dataframe(dataframe, input_labels, output_labels)

        dataset = DSL_Data_Set(is_trace=True)
        if not input_dataframe.empty:
            dataset.input_dataframe= input_dataframe
            dataset.input_labels = input_labels
            dataset.length = input_dataframe.shape[0]
        if not output_dataframe.empty:
            dataset.output_dataframe= output_dataframe
            dataset.output_labels = output_labels
            dataset.length = output_dataframe.shape[0]
        return dataset

    def __get_labels_from_csv_dataframe(self, dataframe):
        """returns the input and output labels from the dataframe gotten from the csv file"""
        labels = list(dataframe.keys())
        input_labels = []
        output_labels = []

        for label in labels:
            if label.startswith('input_'):
                input_labels.append(label.split('input_',1)[1])
            elif label.startswith(' input_'):
                input_labels.append(label.split(' input_',1)[1])
            elif label.startswith('output_'):
                output_labels.append(label.split('output_',1)[1])
            elif label.startswith(' output_'):
                output_labels.append(label.split(' output_',1)[1])
            else:
                raise ValueError("The column names in the csv file have to start with 'input_' or 'output_'")
        
        return input_labels, output_labels

    def __split_and_label_csv_dataframe(self, dataframe, input_labels, output_labels):
        """returns the input and output dataframes from the dataframe gotten from the csv file"""
        input_dataframe = dataframe.iloc[:,:len(input_labels)]
        output_dataframe = dataframe.iloc[:,len(input_labels):]

        for idx, label in enumerate(input_labels):
            input_dataframe.columns.values[idx]=label

        for idx, label in enumerate(output_labels):
            output_dataframe.columns.values[idx]=label

        return input_dataframe, output_dataframe

    def write_trace_dataset_to_csv(self, file_name):
        """write the dataset to a csv file"""
        if not isinstance(file_name, str) and not os.path.isfile(file_name):
            raise ValueError("The file name should be a string and refer to a file")
        if Path(file_name).suffix != ".csv":
            raise ValueError("The file is not a csv file")
        if exists(file_name):
            warnings.warn("The file already exist and you are overwriting it")

        open(file_name, 'w').close()
        for idx, dataset in enumerate(self.datasets):
            dataset.write_dataset_to_csv(file_name, mode='a')

            if idx != len(self.datasets)-1:
                with open(file_name, 'a') as file:
                    file.write('\n\n')

    def __check_for_label_errors(self, incoming_data):
        """check if the labels of the incoming data match up with the existing labels in the trace dataset"""
        if self.input_labels and incoming_data.input_labels and (self.input_labels != incoming_data.input_labels):
            raise ValueError("The incoming input labels and the existing input labels don't match")

        if self.output_labels and incoming_data.output_labels and (self.output_labels != incoming_data.output_labels):
            raise ValueError("The incoming output labels and the existing output labels don't match")
    
    def __check_for_append_errors(self, incoming_data):
        """checks if the incoming data can be appended to the existing data in the trace dataset"""
        if any(self.datasets):
            bool_input = any(self.datasets[0].input_dataframe) == any(incoming_data.input_dataframe)
            bool_output = any(self.datasets[0].output_dataframe) == any(incoming_data.output_dataframe)

            if not(bool_input and bool_output):
                raise ValueError("The incoming data can't be appended to the existing data")
    
    def __set_labels(self, incoming_data):
        """if there are no existing labels this function will set the labels"""
        if not self.input_labels and incoming_data.input_labels:
            self.input_labels = incoming_data.input_labels
        
        if not self.output_labels and incoming_data.output_labels:
            self.output_labels = incoming_data.output_labels


if __name__ == "__main__":
    dataframe_1 = pd.DataFrame({'a': [1,2,3,4,5], 'b': [2,4,6,8,10],'c': [3,6,9,12,15]})
    dataframe_2 = pd.DataFrame({'e': [-1,-2,-3,-4,-5], 'd': [-2,-4,-6,-8,-10]})
    dataset = DSL_Data_Set()
    
    # # dataset.write_dataset_to_csv("test_empty.csv")
    dataset.initialize_with_dataframe(input_dataframe=dataframe_1,output_dataframe=dataframe_2)
    for datapoint in dataset:
        if isinstance(datapoint, DSL_Data_Point):
            print(datapoint)
            print(datapoint.input_dataframe)
            print(datapoint.output_dataframe)
    # print(dataset.input_dataframe)
    # print(dataset.output_dataframe)
    name="refactored_testing_framework/testing_framework_files/test_txt_file.txt"
    print(name[-4:])
    dataframe = pd.read_csv("refactored_testing_framework/testing_framework_files/test_txt_file.txt")
    print('ehy')