import csv
import matplotlib.pyplot as plt
import os

def extract_coordinates_from_csv(file_path):
    coordinates = []
    
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx != 0:
                numbers = [float(x) for x in row]
                if len(numbers) >= 3:
                    coordinates.append((numbers[1], numbers[2]))
    
    return coordinates

def plot_coordinates(coordinates, color_choice, label_choice):
    x_vals, y_vals = zip(*coordinates)
    plt.scatter(x_vals, y_vals, color=color_choice, marker='o', label=label_choice)

    
if __name__ == "__main__":
    plt.figure()

    file_path = str(os.path.join("DSL_truck_trailer_multi_stage_model_run_tansig", "iteration_1", "dataset.csv"))
    # file_path = "DSL_truck_trailer_multi_stage_model_run_tansig\iteration_1\dataset.csv"
    coordinates = extract_coordinates_from_csv(file_path)
    plot_coordinates(coordinates,'blue','generated paths')

    file_path_perfect_paths = "truck_trailer_multi_stage_loop_traces_index_v1_dataset.csv" 
    coordinates_perfect_paths = extract_coordinates_from_csv(file_path_perfect_paths)
    plot_coordinates(coordinates_perfect_paths,'red','perfect paths')

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Scatter Plot of Extracted Coordinates")
    plt.legend()
    plt.grid(True)
    plt.show()


    