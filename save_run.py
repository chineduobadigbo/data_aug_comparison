import json
import os
import traceback
import logging
import sys
import tensorflow as tf
from numba import cuda

def save_model(model, model_path, model_type):
    file_path = model_path+model_type+'.h5'
    model.save(file_path)
    return file_path

def increase_run_num(run_num):
    # increase run_num in file
    with open('run_num.txt',"w") as run_num_file:
        run_num_file.write(str(run_num))

def save_run(run_path, run_dict):
    # save new run file
    with open(run_path+'run_info.json', 'w') as fp:
        json.dump(run_dict, fp, indent=4)

# also clears tensorflow graph
def save_cycle(models, cycle_path, cycle_dict, cycle_num):
    cycle_data_temp = {}
    # add cycle info
    cycle_data_temp['Cycle'] = cycle_dict
    for model in models:
        cycle_data_temp[model.model_name] = model.train_info
    cycle_data = {}
    cycle_data['Cycle_{}'.format(cycle_num)] = cycle_data_temp
    with open(cycle_path+'cycle_info.json', 'w') as fp:
        json.dump(cycle_data, fp, indent=4)
    # cuda.select_device(0)
    # cuda.close()
    # cuda.select_device(0)

def save_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_run_dict(file_path='run.json'):
    with open(file_path, 'r') as json_file:
        json_object = json.load(json_file)
    return dict(json_object)

def read_run_num(file_path='run_num.txt'):
    run_num = []
    try:
        with open(file_path,"r") as run_num_file:
            for line in run_num_file:
                run_num.append(int(line.strip('\n')))
        current_run_num = run_num[0]+1
    except:
        trace = traceback.format_exc()
        logging.error('An error occured while trying to fetch run_num. Error: {}'.format(trace))
        sys.exit(-1)
    return current_run_num