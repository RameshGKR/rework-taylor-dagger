#!/usr/bin/python
import tensorflow as tf

from keras.models import Sequential
from keras import models
import numpy as np
import sys
import yaml

def h5_to_yml(argv, custom_objects_use_case):
    input_filename = argv[0]
    output_filename = argv[1]
    
    model = models.load_model(input_filename, custom_objects=custom_objects_use_case)

    dnn_dict = {}
    dnn_dict['weights'] = {}
    dnn_dict['offsets'] = {}
    dnn_dict['activations'] = {}

    layer_count = 1
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            dnn_dict['weights'][layer_count] = []
            for row in layer.get_weights()[0].T:
                a = []
                try:    
                    for column in row: a.append(float(column))
                except:
                    a.append(float(row))
                dnn_dict['weights'][layer_count].append(a)
            
            dnn_dict['offsets'][layer_count] = []
            for row in layer.get_weights()[1].T:
                dnn_dict['offsets'][layer_count].append(float(row))
            
            if 'normalization' not in str(layer.output):
                if hasattr(layer, 'activation'):
                    if 'sigmoid' in str(layer.activation):
                        dnn_dict['activations'][layer_count] = 'Sigmoid'
                    elif 'tanh' in str(layer.activation):
                        dnn_dict['activations'][layer_count] = 'Tanh'
                    elif 'relu' in str(layer.activation):
                        dnn_dict['activations'][layer_count] = 'Relu'
                    elif 'swish' in str(layer.activation):
                        dnn_dict['activations'][layer_count] = 'Swish'
                    elif 'restricted_output' in str(layer.activation):
                        dnn_dict['activations'][layer_count] = 'Sigmoid'
                    else:
                        dnn_dict['activations'][layer_count] = 'Linear'
                else:
                    dnn_dict['activations'][layer_count] = 'Sigmoid'
            else:
                dnn_dict['activations'][layer_count] = 'Linear'
            layer_count += 1

    #dnn_dict = remove_multiple_output_layers(dnn_dict)

    with open(output_filename, 'w') as f:
        yaml.dump(dnn_dict, f)

def remove_multiple_output_layers(dnn_dict):
    layer_amount = len(dnn_dict['activations'])
    if not dnn_dict['activations'][layer_amount-2] == 'Linear':
        return dnn_dict
    
    new_dnn_dict = {}
    new_dnn_dict['weights'] = {}
    new_dnn_dict['offsets'] = {}
    new_dnn_dict['activations'] = {}

    append_from_now = False
    layer_append_from_now = -1

    for layer_count in range(layer_amount):
        if layer_count == 0 or dnn_dict['activations'][layer_count+1] != 'Linear':
            new_dnn_dict['weights'][layer_count+1] = dnn_dict['weights'][layer_count+1]
            new_dnn_dict['offsets'][layer_count+1] = dnn_dict['offsets'][layer_count+1]
            new_dnn_dict['activations'][layer_count+1] = dnn_dict['activations'][layer_count+1]
        elif not append_from_now:
            append_from_now = True
            layer_append_from_now = layer_count

            new_dnn_dict['weights'][layer_count+1] = dnn_dict['weights'][layer_count+1]
            new_dnn_dict['offsets'][layer_count+1] = dnn_dict['offsets'][layer_count+1]
            new_dnn_dict['activations'][layer_count+1] = dnn_dict['activations'][layer_count+1]
        elif append_from_now:
            new_dnn_dict['weights'][layer_append_from_now+1].append(dnn_dict['weights'][layer_count+1][0])
            new_dnn_dict['offsets'][layer_append_from_now+1].append(dnn_dict['offsets'][layer_count+1][0])
    
    return new_dnn_dict

if __name__ == '__main__':
    h5_to_yml(sys.argv[1:])