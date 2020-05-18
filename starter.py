import subprocess
import time
import argparse
import sys
import logging
from save_run import save_directory, read_run_num, increase_run_num, read_run_dict, save_run
from load_logger import load_logger_conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script runs through the whole training process.')
    # set up logger
    load_logger_conf()
    # num_new_samples is per class
    experiment_cycles = read_run_dict()
    current_run_num = read_run_num()
    increase_run_num(current_run_num)
    # create run number dir
    new_run_path = './runs/run_{}/'.format(current_run_num)
    save_directory(new_run_path)
    start_run_time = time.time()
    for cycle, cycle_dict in experiment_cycles.items():
        cmd = ['python', 'main.py', '--run', str(current_run_num), '--cycle', str(cycle)]
        try:
            proc = subprocess.Popen(cmd)
            proc.wait()
        except Exception as e:
            #print('An error occurred.  Error code:', err.decode())
            logging.error('An error was thrown in the subprocess. Error: {}'.format(e))
            sys.exit(1)
        print('Done with cycle: {}'.format(cycle))
    end_run_time = time.time()
    experiment_cycles['run_time_sec'] = (end_run_time-start_run_time)
    experiment_cycles['run_time_mins'] = ((end_run_time-start_run_time)/60)
    save_run(current_run_num, new_run_path, experiment_cycles)