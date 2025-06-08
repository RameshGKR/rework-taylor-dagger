import os
import math
import numpy as np
import pandas as pd
from DSL_data_classes import DSL_Data_Point, DSL_Data_Set, DSL_Trace_Data_Set
from DSL_policy_case import Policy

from imp_NN_training import Train_NN_Policy_parameters, Retrain_NN_Policy_parameters
from imp_validate_datasets import Validation_parameters
from imp_validate_trace_datasets import Validation_trace_parameters

total_dataset = DSL_Data_Set()

expert_trace_dataset = DSL_Trace_Data_Set()
expert_trace_dataset.initialize_from_csv(str(os.path.join("Tutorial_data", "short_trace_data_set_perfect_paths.csv")))
# expert_trace_dataset.initialize_from_csv("Tutorial_data\short_trace_data_set_perfect_paths.csv")

# indexes = list(range(10,101))
# trace_dataset = DSL_Trace_Data_Set()

# for dataset in expert_trace_dataset:
#     dataset.is_trace = False
#     dataset.remove_datapoints(indexes)
#     dataset.is_trace = True
#     trace_dataset.append_dataset(dataset)

print("hey")