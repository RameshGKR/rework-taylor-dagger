import csv
from typing import List

class Use_Case:
    def __init__(self):
        self.labels_input = False
        self.NN_labels_input = False
        self.labels_output = False
        self.hyperparameters = False
        self.end_iterations = False
        self.timestep = False
        self.perfect_paths = False
        self.custom_objects = False

    def give_hypermodel(self, normalizer, train_features):
        pass

    def give_NN_model(self, normalizer, best_hps, train_features):
        pass

    def give_expert_actions(self, input):
        pass

    def give_next_state(self, current_X, current_U, time_step):
        pass
    
    def give_NN_data(self, data):
        return data
    
    def give_NN_data_row(self, data_row):
        return data_row

class Data_set:
    def __init__(self, input_width: int, output_width: int):
        self.input_width = input_width
        self.output_width = output_width
        self.input_data = give_list_list(self.input_width)
        self.output_data = give_list_list(self.output_width)
        self.length = 0

    def load_data_from_file(self, save_place_data: str, load_only_output: bool = False):
        """load data from a file"""
        data_set = []
        with open(save_place_data) as file_name:
            csvreader = csv.reader(file_name)
            for row in csvreader:
                data_set.append([float(i) for i in row])

        self.load_data_from_row_list(self, data_set, load_only_output = load_only_output)
       
    def load_data_from_col_list(self, col_list: List[List[float]], load_only_output: bool = False):
        """load data from a column list"""
        input_data = []
        output_data = []

        for idx, column in enumerate(col_list):
            if load_only_output:
                idx = idx + self.input_width

            if idx < self.input_width:
                input_data.append(column)
            else:
                output_data.append(column)
        
        if input_data:
            self.input_data = input_data

        if output_data:    
            self.output_data = output_data

        if load_only_output:
            self.length = len(self.output_data[0])
        else:
            self.length = len(self.input_data[0])
    
    def load_data_from_row_list(self, row_list: List[List[float]], load_only_output: bool = False):
        """load data from a row list"""
        input_data = give_list_list(self.input_width)
        output_data = give_list_list(self.output_width)

        self.length = len(row_list)

        for row in row_list:
            for idx, row_parameter in enumerate(row):
                if load_only_output:
                    idx = idx + self.input_width

                if idx < self.input_width:
                    input_data[idx].append(row_parameter)
                else:
                    output_data[idx-self.input_width].append(row_parameter)

        if input_data:
            self.input_data = input_data

        if output_data:    
            self.output_data = output_data

    def add_rows(self, row_list: List[List[float]], load_only_output: bool = False):
        """add rows to the data"""
        for row in row_list:
            for idx, row_parameter in enumerate(row):
                if load_only_output:
                    idx = idx + self.input_width

                if idx < self.input_width:
                    self.input_data[idx].append(row_parameter)
                else:
                    self.output_data[idx-self.input_width].append(row_parameter)
            
            self.length = self.length + 1

    def give_columns(self, give_input: bool, give_output: bool):
        """gives the data in collumns"""
        data_columns = []

        if give_input:
            data_columns = data_columns + self.input_data

        if give_output:
            data_columns = data_columns + self.output_data
        
        return data_columns 

    def give_rows(self, give_input: bool, give_output: bool):
        """gives the data in rows"""
        data_rows = []

        for idx in range(self.length):
            data_row = []
            if give_input:
                input_row = self.give_input_row(idx)
                data_row = data_row + input_row

            if give_output:
                output_row = self.give_ouput_row(idx)
                data_row = data_row + output_row
            
            data_rows.append(data_row)
        
        return data_rows

    def give_input_row(self, idx: int):
        """gives the input row"""
        input_row = []

        for input_parameter_list in self.input_data:
            input_row.append(input_parameter_list[idx])

        return input_row

    def give_ouput_row(self, idx: int):
        """gives the output row"""
        output_row = []

        for output_parameter_list in self.output_data:
            output_row.append(output_parameter_list[idx])

        return output_row

    def save_data(self, save_place: str):
        """saves the data that is generated"""
        data_rows = self.give_rows(give_input=True, give_output=True)
        with open(save_place, "a+") as output:
            writer = csv.writer(output, lineterminator='\n')
            for datarow in data_rows:
                writer.writerow(datarow)
    
    def save_data_with_index(self, save_place: str, start_index: int):
        """saves the data that is generated"""
        data_rows = self.give_rows(give_input=True, give_output=True)

        with open(save_place+"_index", "a+") as output:
            writer = csv.writer(output, lineterminator='\n')
            for idx, datarow in enumerate(data_rows):
                datarow_with_indexs=[]
                datarow_with_indexs.append(start_index)
                datarow_with_indexs.append(idx+1)
                for data in datarow:
                    datarow_with_indexs.append(data.copy())
                writer.writerow(datarow_with_indexs)

    def delete_False_outputs(self):
        """deletes the input and output row if the output is False"""
        for idx in reversed(range(self.length)):
            if not self.output_data[0][idx]:
                for input_col in self.input_data:
                    input_col.pop(idx)


                for output_col in self.output_data:
                    output_col.pop(idx)

        self.length = len(self.output_data[0])

def give_list_list(amount: int) -> List[List]:
    """gives a list of empty lists"""
    list_list = []

    for _ in range(amount):
        list_list.append([])

    return list_list