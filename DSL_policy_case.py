import numbers
import pandas as pd
import random
from numpy import round
from DSL_data_classes import DSL_Data_Point, DSL_Data_Set, DSL_Trace_Data_Set

class Policy:
    def __init__(self, output_function):
        """initialize the policy class object with the output function of the policy"""
        self.output_function = output_function

    def __add__(self, object):
        """helper function to add policy with other policies or numbers the policy is first in order"""
        if not callable(self.output_function):
            raise ValueError("Error: self output function has to be a function")
        function_1 = self.give_output

        if isinstance(object, Policy):
            if not callable(object.output_function):
                raise ValueError("Error: object output function has to be a function")
            
            function_2 = object.give_output
            def changed_function(input):
                output_1 = function_1(input)
                output_2 = function_2(input)
                output = Policy_add_policy(output_1, output_2)
                return output
        elif isinstance(object, numbers.Number):
            def changed_function(input):
                output_1 = function_1(input)
                output = Policy_add_number(output_1, object)
                return output
        else:
            raise ValueError("Error: This object can't be added with a policy")

        return Policy(changed_function)
    def __radd__(self, object):
        """helper function to add policy with other numbers when the policy is second in order"""
        if not callable(self.output_function):
            raise ValueError("Error: self output function has to be a function")
        function_1 = self.give_output

        if isinstance(object, numbers.Number):
            def changed_function(input):
                output_1 = function_1(input)
                output = Policy_add_number(output_1, object)
                return output
        else:
            raise ValueError("Error: This object can't be added with a policy")

        return Policy(changed_function)

    def __sub__(self, object):
        """helper function to subtract policy with other policies or numbers the policy is first in order"""
        if not callable(self.output_function):
            raise ValueError("Error: self output function has to be a function")
        function_1 = self.give_output

        if isinstance(object, Policy):
            if not callable(object.output_function):
                raise ValueError("Error: object output function has to be a function")
            
            function_2 = object.give_output
            def changed_function(input):
                output_1 = function_1(input)
                output_2 = function_2(input)
                output = Policy_subtract_policy(output_1, output_2)
                return output
        elif isinstance(object, numbers.Number):
            def changed_function(input):
                output_1 = function_1(input)
                output = Policy_subtract_number(output_1, object)
                return output
        else:
            raise ValueError("Error: This object can't be added with a policy")

        return Policy(changed_function)
    def __rsub__(self, object):
        """helper function to subtract policy with other numbers when the policy is second in order"""
        if not callable(self.output_function):
            raise ValueError("Error: self output function has to be a function")
        function_1 = self.give_output

        if isinstance(object, numbers.Number):
            def changed_function(input):
                output_1 = function_1(input)
                output = Policy_subtract_number(output_1, object, True)
                return output
        else:
            raise ValueError("Error: This object can't be added with a policy")

        return Policy(changed_function)

    def __mul__(self, object):
        """helper function to multiply policy with other policies or numbers the policy is first in order"""
        if not callable(self.output_function):
            raise ValueError("Error: self output function has to be a function")
        function_1 = self.give_output
        
        if isinstance(object, Policy):
            if not callable(object.output_function):
                raise ValueError("Error: object output function has to be a function")
            
            function_2 = object.give_output
            def changed_function(input):
                output_1 = function_1(input)
                output_2 = function_2(input)
                output = Policy_multiply_policy(output_1, output_2)
                return output
        elif isinstance(object, numbers.Number):
            def changed_function(input):
                output_1 = function_1(input)
                output = Policy_multiply_number(output_1, object)
                return output
        else:
            raise ValueError("Error: This object can't be added with a policy")

        return Policy(changed_function)
    def __rmul__(self, object):
        """helper function to multiply policy with other numbers when the policy is second in order"""
        if not callable(self.output_function):
            raise ValueError("Error: self output function has to be a function")
        function_1 = self.give_output
        
        if isinstance(object, numbers.Number):
            def changed_function(input):
                output_1 = function_1(input)
                output = Policy_multiply_number(output_1, object)
                return output
        else:
            raise ValueError("Error: This object can't be added with a policy")
        
        return Policy(changed_function)

    def __truediv__(self, object):
        """helper function to divide policy with other policies or numbers the policy is first in order"""
        if not callable(self.output_function):
            raise ValueError("Error: self output function has to be a function")
        function_1 = self.give_output
        
        if isinstance(object, Policy):
            if not callable(object.output_function):
                raise ValueError("Error: object output function has to be a function")
            
            function_2 = object.give_output
            def changed_function(input):
                output_1 = function_1(input)
                output_2 = function_2(input)
                output = Policy_divide_policy(output_1, output_2)
                return output
        elif isinstance(object, numbers.Number):
            def changed_function(input):
                output_1 = function_1(input)
                output = Policy_divide_number(output_1, object)
                return output
        else:
            raise ValueError("Error: This object can't be added with a policy")

        return Policy(changed_function)
    def __rtruediv__(self, object):
        """helper function to divide policy with other numbers when the policy is second in order"""
        if not callable(self.output_function):
            raise ValueError("Error: self output function has to be a function")
        function_1 = self.give_output
        
        if isinstance(object, numbers.Number):
            def changed_function(input):
                output_1 = function_1(input)
                output = Policy_divide_number(output_1, object, True)
                return output
        else:
            raise ValueError("Error: This object can't be added with a policy")

        return Policy(changed_function)

    def statistical_combination(self, other_policies_list, weights_list):
        if not callable(self.output_function):
            raise ValueError("Error: self output function has to be a function")
        if not isinstance(weights_list, list):
            raise ValueError("Error: the weight_list has to be a list")
        if len(weights_list)!=len(other_policies_list)+1:
            raise ValueError("Error: there have to be as much weights as policies including the policy running the function")
        if not isinstance(other_policies_list, list):
            raise ValueError("Error: the other_policies_list has to be a list")
        for policy in other_policies_list:
            if not callable(policy.output_function):
                raise ValueError("Error: all policy output functions in the other_policy_list have to be functions")
        
        function_1 = self.give_output

        def changed_function(input):
            random_number = random.random()
            total=sum(weights_list)
            weight_limit=random_number*total

            running_weight=0
            random_index=0
            for weight in weights_list:
                if running_weight<weight_limit and weight_limit<running_weight+weight:
                    break
                else:
                    running_weight=running_weight+weight
                    random_index=random_index+1

            if random_index == 0:
                output = function_1(input)
            else:
                output = other_policies_list[random_index-1].give_output(input)

            return output
        
        return Policy(changed_function)

    def give_output(self, input):
        """is wrapper for the output function"""
        if not (isinstance(input, DSL_Data_Set) or isinstance(input, DSL_Data_Point)):
            raise ValueError("Error: input has to be a DSL_Data_Set or a DSL_Data_Point")
        if not input.output_dataframe.empty:
            raise ValueError("Error: the dataset or datapoint given can not have an output")
        if not callable(self.output_function):
            raise ValueError("Error: output function has to be a function")
        
        if isinstance(input, DSL_Data_Set):
            output_dataset = DSL_Data_Set()

            for idx, input_datapoint in enumerate(input):
                if idx%100==0:
                    print(str(int(round((idx+1)/100)))+"/"+str(int(round(input.length/100))))
                output_datapoint = self.output_function(input_datapoint)

                if not isinstance(output_datapoint, DSL_Data_Point):
                    raise ValueError("Error: output of the output function has to be a DSL_Data_Point")
                output_dataset.append_datapoint(output_datapoint)
            
            output = output_dataset
        elif isinstance(input, DSL_Data_Point):
            output_datapoint = self.output_function(input)

            if not isinstance(output_datapoint, DSL_Data_Point):
                raise ValueError("Error: output of the output function has to be a DSL_Data_Point")
            
            output = output_datapoint
        
        return output

def Policy_add_policy(datapoint_1, datapoint_2):
    input_dataframe = datapoint_1.input_dataframe
    new_output_dataframe = pd.DataFrame()

    output_dataframe_1 = datapoint_1.output_dataframe
    output_dataframe_2 = datapoint_2.output_dataframe

    for label in output_dataframe_1.columns:
        element1 = output_dataframe_1[label][0]
        element2 = output_dataframe_2[label][0]

        new_element = element1+element2
        new_output_dataframe[label]=[new_element]

    new_output=DSL_Data_Point()
    new_output.initialize_with_dataframe(input=input_dataframe, output=new_output_dataframe)

    return new_output

def Policy_add_number(datapoint_1, number):
    input_dataframe = datapoint_1.input_dataframe
    new_output_dataframe = pd.DataFrame()

    output_dataframe_1 = datapoint_1.output_dataframe

    for label in output_dataframe_1.columns:
        element1 = output_dataframe_1[label][0]
        
        new_element = element1+number
        new_output_dataframe[label]=[new_element]

    new_output=DSL_Data_Point()
    new_output.initialize_with_dataframe(input=input_dataframe, output=new_output_dataframe)

    return new_output

def Policy_subtract_policy(datapoint_1, datapoint_2):
    input_dataframe = datapoint_1.input_dataframe
    new_output_dataframe = pd.DataFrame()

    output_dataframe_1 = datapoint_1.output_dataframe
    output_dataframe_2 = datapoint_2.output_dataframe

    for label in output_dataframe_1.columns:
        element1 = output_dataframe_1[label][0]
        element2 = output_dataframe_2[label][0]

        new_element = element1-element2
        new_output_dataframe[label]=[new_element]

    new_output=DSL_Data_Point()
    new_output.initialize_with_dataframe(input=input_dataframe, output=new_output_dataframe)

    return new_output

def Policy_subtract_number(datapoint_1, number, switch=False):
    input_dataframe = datapoint_1.input_dataframe
    new_output_dataframe = pd.DataFrame()

    output_dataframe_1 = datapoint_1.output_dataframe

    for label in output_dataframe_1.columns:
        element1 = output_dataframe_1[label][0]
        
        if switch:
            new_element = number-element1
        else:
            new_element = element1-number
        new_output_dataframe[label]=[new_element]

    new_output=DSL_Data_Point()
    new_output.initialize_with_dataframe(input=input_dataframe, output=new_output_dataframe)

    return new_output

def Policy_multiply_policy(datapoint_1, datapoint_2):
    input_dataframe = datapoint_1.input_dataframe
    new_output_dataframe = pd.DataFrame()

    output_dataframe_1 = datapoint_1.output_dataframe
    output_dataframe_2 = datapoint_2.output_dataframe

    for label in output_dataframe_1.columns:
        element1 = output_dataframe_1[label][0]
        element2 = output_dataframe_2[label][0]

        new_element = element1*element2
        new_output_dataframe[label]=[new_element]

    new_output=DSL_Data_Point()
    new_output.initialize_with_dataframe(input=input_dataframe, output=new_output_dataframe)

    return new_output

def Policy_multiply_number(datapoint_1, number):
    input_dataframe = datapoint_1.input_dataframe
    new_output_dataframe = pd.DataFrame()

    output_dataframe_1 = datapoint_1.output_dataframe

    for label in output_dataframe_1.columns:
        element1 = output_dataframe_1[label][0]
        
        new_element = element1*number
        new_output_dataframe[label]=[new_element]

    new_output=DSL_Data_Point()
    new_output.initialize_with_dataframe(input=input_dataframe, output=new_output_dataframe)

    return new_output

def Policy_divide_policy(datapoint_1, datapoint_2):
    input_dataframe = datapoint_1.input_dataframe
    new_output_dataframe = pd.DataFrame()

    output_dataframe_1 = datapoint_1.output_dataframe
    output_dataframe_2 = datapoint_2.output_dataframe

    for label in output_dataframe_1.columns:
        element1 = output_dataframe_1[label][0]
        element2 = output_dataframe_2[label][0]

        new_element = element1/element2
        new_output_dataframe[label]=[new_element]

    new_output=DSL_Data_Point()
    new_output.initialize_with_dataframe(input=input_dataframe, output=new_output_dataframe)

    return new_output

def Policy_divide_number(datapoint_1, number, switch=False):
    input_dataframe = datapoint_1.input_dataframe
    new_output_dataframe = pd.DataFrame()

    output_dataframe_1 = datapoint_1.output_dataframe

    for label in output_dataframe_1.columns:
        element1 = output_dataframe_1[label][0]
        
        if switch:
            new_element = number/element1
        else:
            new_element = element1/number
        new_output_dataframe[label]=[new_element]

    new_output=DSL_Data_Point()
    new_output.initialize_with_dataframe(input=input_dataframe, output=new_output_dataframe)

    return new_output
