import csv
import yaml
from numpy import pi, cos, sin, tan

trace_length = 50

with open('truck_trailer_para.yaml', 'r') as file:
    para = yaml.safe_load(file)

L0 = para['truck']['L']
M0 = para['truck']['M']
W0 = para['truck']['W']
L1 = para['trailer1']['L']
M1 = para['trailer1']['M']
W1 = para['trailer1']['W']

with open("truck_trailer_multi_stage_loop_traces.csv") as file_name:
    csvreader = csv.reader(file_name)
    MPC_theta1 = []
    for idx, row in enumerate(csvreader):
        index=idx%trace_length
        x1 = float(row[0])
        y1 = float(row[1])
        theta0 = float(row[2])
        theta1 = float(row[3])
        v0 = float(row[4])
        dtheta0 = float(row[5])

        # beta01 = theta0 - theta1
        # v1 = v0*cos(beta01) + M0*sin(beta01)*dtheta0

        with open("truck_trailer_multi_stage_loop_traces_index.csv", "a+") as output_file:
            writer = csv.writer(output_file, lineterminator='\n')
            writer.writerow([index,x1,y1,theta0,theta1,v0,dtheta0])

with open("truck_trailer_multi_stage_loop_traces_index.csv") as file_name:
    csvreader = csv.reader(file_name)
    MPC_theta1 = []
    for idx,row in enumerate(csvreader):
        if idx%trace_length==0:
            with open('truck_trailer_multi_stage_loop_traces_index_start_points.csv', "a+") as output_file:
                writer = csv.writer(output_file, lineterminator='\n')
                writer.writerow(row[0:5])
        


with open("truck_trailer_multi_stage_loop_traces_index.csv") as file_name:
    csvreader = csv.reader(file_name)
    MPC_theta1 = []
    for idx,row in enumerate(csvreader):
        if idx%trace_length==0:
            with open('truck_trailer_multi_stage_loop_traces_index_traces.csv', "a+") as output_file:
                writer = csv.writer(output_file, lineterminator='\n')
                writer.writerow([])
                writer.writerow([])
                writer.writerow(["input_index","input_x1","input_y1","input_theta0","input_theta1","output_v0","output_delta0"])

        
        with open('truck_trailer_multi_stage_loop_traces_index_traces.csv', "a+") as output_file:
                writer = csv.writer(output_file, lineterminator='\n')
                writer.writerow(row)


