from DSL_data_classes import DSL_Data_Point, DSL_Data_Set, DSL_Trace_Data_Set
from DSL_policy_case import Policy
import pandas as pd

class Train_policy:
    def __init__(self, training_function=False, general_training_parameters=False):
        """Initialize a train policy object with the training function and general training parameters. general training parameters are parameters that don't change once the train policy object is made"""
        self.training_function = training_function
        self.general_training_parameters = general_training_parameters
    
    def train_policy(self, dataset, training_parameters=False):
        """This function is a wrapper for the training function it takes the training parameters. Training parameters are parameters that can for different calls of the train policy function."""
        if not isinstance(dataset, DSL_Data_Set):
            raise ValueError("input for train policy has to be a dataset")
        if dataset.input_dataframe.empty or dataset.output_dataframe.empty:
            raise ValueError("dataset for train policy has to be have input and output")

        policy = self.training_function(dataset, self.general_training_parameters, training_parameters)

        if not isinstance(policy, Policy):
            raise ValueError("output for training function has to be a policy")
        
        return policy

class Simulate_system:
    def __init__(self, simulate_function=False, general_simulation_parameters=False):
        """Initialize a simulate system object with the simulate function and general simulation parameters. general simulation parameters are parameters that don't change once the simulate system object is made"""
        self.simulate_function = simulate_function
        self.general_simulation_parameters = general_simulation_parameters
    
    def simulate_system(self, input, simulation_parameters=False):
        """This function is a wrapper for the simulate function it takes the simulation parameters. Simulation parameters are parameters that can for different calls of the simulate system function."""
        if not (isinstance(input, DSL_Data_Set) or isinstance(input, DSL_Data_Point)):
            raise ValueError("input for simulate system has to be a dataset or a datpoint")
        if input.input_dataframe.empty or input.output_dataframe.empty:
            raise ValueError("dataset or datapoint for simulate system has to have input and output")
        
        if isinstance(input, DSL_Data_Set):
            output_dataset = DSL_Data_Set()

            for input_datapoint in input:
                output_datapoint = self.simulate_function(input_datapoint, self.general_simulation_parameters, simulation_parameters)

                if not isinstance(output_datapoint, DSL_Data_Point):
                    raise ValueError("output from simulate function has to be a datapoint")
                if  output_datapoint.input_dataframe.empty or not output_datapoint.output_dataframe.empty:
                    raise ValueError("output datapoint from simulate function has to have only input")
                if not output_datapoint.input_labels == input_datapoint.input_labels:
                    raise ValueError("output datapoint input lables from simulate function has to be the same as the input labels of the input datapoint")

                output_dataset.append_datapoint(output_datapoint)
            
            output = output_dataset
        elif isinstance(input, DSL_Data_Point):
            output_datapoint = self.simulate_function(input, self.general_simulation_parameters, simulation_parameters)

            if not isinstance(output_datapoint, DSL_Data_Point):
                raise ValueError("output from simulate function has to be a datapoint")
            if  output_datapoint.input_dataframe.empty or not output_datapoint.output_dataframe.empty:
                raise ValueError("output datapoint from simulate function has to have only input")
            if not output_datapoint.input_labels == input.input_labels:
                raise ValueError("output datapoint input lables from simulate function has to be the same as the input labels of the input datapoint")
            
            output = output_datapoint


        return output

    def simulate_system_traces(self, policy: Policy, startpoints_dataset: DSL_Data_Set, amount_of_steps: int, simulation_parameters=False) -> DSL_Trace_Data_Set:
        """This function is a will simulate traces it will use the datapoints in the startpoints dataset to start the traces and then it will use the simulate function to find the next datapoint input
          and use the policy to finde the datapoint output this will continue for the pre determined amount of steps."""
        if not isinstance(policy, Policy):
            raise ValueError("policy for simulate system traces has to be a policy")
        if not isinstance(startpoints_dataset, DSL_Data_Set):
            raise ValueError("startpoints_dataset for simulate system traces has to be a dataset")
        if startpoints_dataset.input_dataframe.empty or not startpoints_dataset.output_dataframe.empty:
            raise ValueError("dataset for simulate system traces has to have no empty input and a empty output")
        if not isinstance(amount_of_steps, int):
            raise ValueError("amount_of_steps for simulate system traces has to be a int")
        
        trace_data_set = DSL_Trace_Data_Set()

        startpoints_dataset.output = policy.give_output(startpoints_dataset.input)

        for startpoint_datapoint in startpoints_dataset:
            dataset=DSL_Data_Set(is_trace=True)
            dataset.append_datapoint(startpoint_datapoint)
            trace_data_set.append_dataset(dataset)

        previous_point_dataset = startpoints_dataset
        
        for idx in range(amount_of_steps):
            print(str(idx+1)+"/"+str(amount_of_steps))
            next_point_dataset = self.simulate_system.simulate_system(previous_point_dataset, simulation_parameters)
            next_point_dataset.output = policy.give_output(next_point_dataset.input)

            trace_data_set.append_dataset_pointwise(next_point_dataset)
            previous_point_dataset = next_point_dataset
        
        return trace_data_set
    
class Simulate_system_traces:
    def __init__(self, simulate_function=False, general_simulation_parameters=False):
        """Initialize a simulate system traces object with the simulate function and general simulation parameters. general simulation parameters are parameters that don't change once the simulate system traces object is made"""
        self.simulate_system = Simulate_system(simulate_function, general_simulation_parameters)
        self.general_simulation_parameters = general_simulation_parameters
    
    def simulate_system_traces(self, policy: Policy, startpoints_dataset: DSL_Data_Set, amount_of_steps: int, simulation_parameters=False) -> DSL_Trace_Data_Set:
        """This function is a will simulate traces it will use the datapoints in the startpoints dataset to start the traces and then it will use the simulate function to find the next datapoint input
          and use the policy to finde the datapoint output this will continue for the pre determined amount of steps."""
        if not isinstance(policy, Policy):
            raise ValueError("policy for simulate system traces has to be a policy")
        if not isinstance(startpoints_dataset, DSL_Data_Set):
            raise ValueError("startpoints_dataset for simulate system traces has to be a dataset")
        if startpoints_dataset.input_dataframe.empty or not startpoints_dataset.output_dataframe.empty:
            raise ValueError("dataset for simulate system traces has to have no empty input and an empty output")
        if not isinstance(amount_of_steps, int):
            raise ValueError("amount_of_steps for simulate system traces has to be a int")
        
        trace_data_set = DSL_Trace_Data_Set()

        startpoints_dataset.output = policy.give_output(startpoints_dataset.input)

        for startpoint_datapoint in startpoints_dataset:
            dataset=DSL_Data_Set(is_trace=True)
            dataset.append_datapoint(startpoint_datapoint)
            trace_data_set.append_dataset(dataset)

        previous_point_dataset = startpoints_dataset
        
        for idx in range(amount_of_steps):
            print(str(idx+1)+"/"+str(amount_of_steps))
            next_point_dataset = self.simulate_system.simulate_system(previous_point_dataset, simulation_parameters)
            
            output_dataset = policy.give_output(next_point_dataset.input)
            if output_dataset.error_included:
                break
            next_point_dataset.output = output_dataset


            trace_data_set.append_dataset_pointwise(next_point_dataset)
            previous_point_dataset = next_point_dataset
        
        return trace_data_set

class Validate_datasets:
    def __init__(self, validation_function=False, general_validation_parameters=False):
        """Initialize a validate datasets object with the validation function and general validation parameters. general validation parameters are parameters that don't change once the validate datasets object is made"""
        self.validation_function = validation_function
        self.general_validation_parameters = general_validation_parameters
    
    def validate_datasets(self, dataset_1, dataset_2, validation_parameters=False):
        """This function is a wrapper for the validation function it takes the validation parameters. Validation parameters are parameters that can for different calls of the validate datasets function."""
        if not isinstance(dataset_1, DSL_Data_Set) or not isinstance(dataset_2, DSL_Data_Set):
            raise ValueError("input for validate system has to be a dataset")
        if dataset_1.length != dataset_2.length:
            raise ValueError("the datasets are not the same length")
        if dataset_1.input_labels != dataset_2.input_labels:
            raise ValueError("the input labels of the datasets have to be the same")
        if dataset_1.output_labels != dataset_2.output_labels:
            raise ValueError("the output labels of the datasets have to be the same")
        if dataset_1.input_dataframe.empty or dataset_1.output_dataframe.empty or dataset_2.input_dataframe.empty or dataset_2.output_dataframe.empty:
            raise ValueError("the datasets have to have input and output")

        output = self.validation_function(dataset_1, dataset_2, self.general_validation_parameters, validation_parameters)

        return output

class Validate_trace_datasets:
    def __init__(self, trace_validation_function=False, general_validation_trace_parameters=False):
        """Initialize a validate trace datasets object with the trace validation function and general validation trace parameters. general validation trace parameters are parameters that don't change once the validate trace datasets object is made"""
        self.trace_validation_function = trace_validation_function
        self.general_validation_parameters = general_validation_trace_parameters
    
    def validate_trace_datasets(self, trace_dataset_1, trace_dataset_2, validation_trace_parameters=False):
        """This function is a wrapper for the trace validation function it takes the validation trace parameters. Validation trace parameters are parameters that can for different calls of the validate trace datasets function."""
        if not isinstance(trace_dataset_1, DSL_Trace_Data_Set) or not isinstance(trace_dataset_2, DSL_Trace_Data_Set):
            raise ValueError("input for validate system has to be a dataset")
        if len(trace_dataset_1.datasets) != len(trace_dataset_2.datasets):
            raise ValueError("the tracedatasets do not have the same amount of datasets")
        if trace_dataset_1.datasets[0].length != trace_dataset_2.datasets[0].length:
            raise ValueError("the datasets are not the same length")
        if trace_dataset_1.input_labels != trace_dataset_2.input_labels:
            raise ValueError("the input labels of the datasets have to be the same")
        if trace_dataset_1.output_labels != trace_dataset_2.output_labels:
            raise ValueError("the output labels of the datasets have to be the same")
        if trace_dataset_1.datasets[0].input_dataframe.empty or trace_dataset_1.datasets[0].output_dataframe.empty or trace_dataset_2.datasets[0].input_dataframe.empty or trace_dataset_2.datasets[0].output_dataframe.empty:
            raise ValueError("the datasets have to have input and output")

        output = self.trace_validation_function(trace_dataset_1, trace_dataset_2, self.general_validation_parameters, validation_trace_parameters)

        return output
    
def test_simulate_function(datapoint):
    input = datapoint.input_dataframe['a'].iloc[0]
    output = datapoint.output_dataframe['f'].iloc[0]

    output_datapoint = DSL_Data_Point(input={'a':input+output})
    return output_datapoint

def test_policy_function(datapoint):
    input = datapoint.input_dataframe['a'].iloc[0]

    dataframe = pd.DataFrame({'f':input}, index=[0])

    output_datapoint = DSL_Data_Point()
    output_datapoint.set_datapoint_output_with_dataframe(dataframe)
    return output_datapoint


if __name__ == "__main__":
    startpointdataset = DSL_Data_Set()
    dataframe_1 = pd.DataFrame({'a': [-5,-4,-3,-2,-1,0,1,2,3,4,5]})
    startpointdataset.initialize_with_dataframe(input_dataframe=dataframe_1)

    policy = Policy(test_policy_function)
    simulatesystemtraces = Simulate_system_traces(test_simulate_function)

    trace_dataset = simulatesystemtraces.simulate_system_traces(policy=policy, startpoints_dataset=startpointdataset, amount_of_steps=10)

    for dataset in trace_dataset:
        print(dataset.input_dataframe)
        print(dataset.output_dataframe)
