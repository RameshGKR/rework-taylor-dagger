import pandas as pd
from casadi import vertcat
from truck_trailer_multistage_loop_expert_function import get_expert_action_truck_trailer

# Load the CSV
df = pd.read_csv("call_expert.csv")

# Prepare output columns
output_v0_list = []
output_delta0_list = []

# Loop through each row
for _, row in df.iterrows():
    input_vector = vertcat(row['input_theta1'], row['input_x1'], row['input_y1'], row['input_theta0'])
    index = int(row['input_index'])
    output = get_expert_action_truck_trailer(input_vector, index)
    output_v0_list.append(float(output[0]))
    output_delta0_list.append(float(output[1]))

# Add to DataFrame
df['output_v0'] = output_v0_list
df['output_delta0'] = output_delta0_list

# Save to new CSV
df.to_csv("call_expert_filled.csv", index=False)
print("MPC outputs written to call_expert_filled.csv")
