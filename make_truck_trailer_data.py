import csv
import numpy as np

with open("truck_trailer_run_test_2.csv") as file_name:
    csvreader = csv.reader(file_name)
    theta1_ar = []
    x1_ar = []
    y1_ar = []
    theta0_ar = []

    i=1
    for row in csvreader:

        if (i-1)%101==0:
            print("begin"+str(i))
            theta1_ar = []
            x1_ar = []
            y1_ar = []
            theta0_ar = []

        theta1_ar.append(float(row[0]))
        x1_ar.append(float(row[1]))
        y1_ar.append(float(row[2]))
        theta0_ar.append(float(row[3]))
        
        if (i-1)%101==100:
            with open('truck_trailer_theta1_perfect_paths.csv', "a+") as output_file:
                writer = csv.writer(output_file, lineterminator='\n')
                writer.writerow(theta1_ar)
            with open('truck_trailer_x1_perfect_paths.csv', "a+") as output_file:
                writer = csv.writer(output_file, lineterminator='\n')
                writer.writerow(x1_ar)
            with open('truck_trailer_y1_perfect_paths.csv', "a+") as output_file:
                writer = csv.writer(output_file, lineterminator='\n')
                writer.writerow(y1_ar)
            with open('truck_trailer_theta0_perfect_paths.csv', "a+") as output_file:
                writer = csv.writer(output_file, lineterminator='\n')
                writer.writerow(theta0_ar)
            

        # with open('truck_trailer_start_dataset.csv', "a+") as output_file:
        #         writer = csv.writer(output_file, lineterminator='\n')
        #         writer.writerow(row)
        i=i+1


