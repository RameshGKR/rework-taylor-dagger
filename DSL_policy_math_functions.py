import pandas as pd
from DSL_data_classes import DSL_Data_Point

def test_policy_add_functions():
    input_dataframe = pd.DataFrame({'g': [5], 'h': [6], 'j': [7]})
    dataframe_1 = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
    dataframe_2 = pd.DataFrame({'a': [-2], 'b': [-4], 'c': [-6]})

    datapoint_1 = DSL_Data_Point()
    datapoint_1.set_datapoint_input_with_dataframe(input_dataframe)
    datapoint_1.set_datapoint_output_with_dataframe(dataframe_1)

    datapoint_2 = DSL_Data_Point()
    datapoint_2.set_datapoint_input_with_dataframe(input_dataframe)
    datapoint_2.set_datapoint_output_with_dataframe(dataframe_2)

    output_datapoint_1 = Policy_add_policy(datapoint_1, datapoint_2)
    output_datapoint_2 = Policy_add_number(datapoint_1, 5.21)
    print(output_datapoint_1.input_dataframe.values)
    print(output_datapoint_1.output_dataframe.values)
    print(output_datapoint_2.input_dataframe.values)
    print(output_datapoint_2.output_dataframe.values)

def test_policy_subtract_functions():
    input_dataframe = pd.DataFrame({'g': [5], 'h': [6], 'j': [7]})
    dataframe_1 = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
    dataframe_2 = pd.DataFrame({'a': [-2], 'b': [-4], 'c': [-6]})

    datapoint_1 = DSL_Data_Point()
    datapoint_1.set_datapoint_input_with_dataframe(input_dataframe)
    datapoint_1.set_datapoint_output_with_dataframe(dataframe_1)

    datapoint_2 = DSL_Data_Point()
    datapoint_2.set_datapoint_input_with_dataframe(input_dataframe)
    datapoint_2.set_datapoint_output_with_dataframe(dataframe_2)

    output_datapoint_1 = Policy_subtract_policy(datapoint_1, datapoint_2)
    output_datapoint_2 = Policy_subtract_number(datapoint_1, 5.21)
    output_datapoint_3 = Policy_subtract_number(datapoint_1, 5.21, True)
    print(output_datapoint_1.input_dataframe.values)
    print(output_datapoint_1.output_dataframe.values)
    print(output_datapoint_2.input_dataframe.values)
    print(output_datapoint_2.output_dataframe.values)
    print(output_datapoint_3.input_dataframe.values)
    print(output_datapoint_3.output_dataframe.values)

def test_policy_multiply_functions():
    input_dataframe = pd.DataFrame({'g': [5], 'h': [6], 'j': [7]})
    dataframe_1 = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
    dataframe_2 = pd.DataFrame({'a': [-2], 'b': [-4], 'c': [-6]})

    datapoint_1 = DSL_Data_Point()
    datapoint_1.set_datapoint_input_with_dataframe(input_dataframe)
    datapoint_1.set_datapoint_output_with_dataframe(dataframe_1)

    datapoint_2 = DSL_Data_Point()
    datapoint_2.set_datapoint_input_with_dataframe(input_dataframe)
    datapoint_2.set_datapoint_output_with_dataframe(dataframe_2)

    output_datapoint_1 = Policy_multiply_policy(datapoint_1, datapoint_2)
    output_datapoint_2 = Policy_multiply_number(datapoint_1, 5.21)
    print(output_datapoint_1.input_dataframe.values)
    print(output_datapoint_1.output_dataframe.values)
    print(output_datapoint_2.input_dataframe.values)
    print(output_datapoint_2.output_dataframe.values)

def test_policy_divide_functions():
    input_dataframe = pd.DataFrame({'g': [5], 'h': [6], 'j': [7]})
    dataframe_1 = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
    dataframe_2 = pd.DataFrame({'a': [-2], 'b': [-4], 'c': [-6]})

    datapoint_1 = DSL_Data_Point()
    datapoint_1.set_datapoint_input_with_dataframe(input_dataframe)
    datapoint_1.set_datapoint_output_with_dataframe(dataframe_1)

    datapoint_2 = DSL_Data_Point()
    datapoint_2.set_datapoint_input_with_dataframe(input_dataframe)
    datapoint_2.set_datapoint_output_with_dataframe(dataframe_2)

    output_datapoint_1 = Policy_divide_policy(datapoint_1, datapoint_2)
    output_datapoint_2 = Policy_divide_number(datapoint_1, 5.21)
    output_datapoint_3 = Policy_divide_number(datapoint_1, 5.21, True)
    print(output_datapoint_1.input_dataframe.values)
    print(output_datapoint_1.output_dataframe.values)
    print(output_datapoint_2.input_dataframe.values)
    print(output_datapoint_2.output_dataframe.values)
    print(output_datapoint_3.input_dataframe.values)
    print(output_datapoint_3.output_dataframe.values)

if __name__ == "__main__":
    test_policy_add_functions()
    test_policy_subtract_functions()
    test_policy_multiply_functions()
    test_policy_divide_functions()
    print('hey')