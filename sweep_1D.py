import numpy
import io, os, json
import concurrent.futures
import subprocess
import copy
import shutil
import pickle

from datetime import datetime
from datetime import date

def sweep_1D(N, d, num_random_graphs, num_init_states, p_noise_list):

    max_threads = 5

    # Setting up directory to save
    today = date.today()
    mmdd =  today.strftime("%m%d%y")[:4]

    data_folder_path = os.path.join('./../vqa_data', mmdd)

    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)

    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    result_save_path = os.path.join(data_folder_path, now)
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers = max_threads) as executor:

        for graph_num in range(num_random_graphs):

            for init_num in range(num_init_states):

                for p in p_noise_list:

                        executor.submit(submit_simulation, str(N), str(d), str(p), str(graph_num), str(init_num), result_save_path)

def submit_simulation(N, d, p, graph_num, init_num, result_save_path):
    '''
    Submit a simulation
    '''

    print('Submitting simulation with N: ', N, ', d: ', d, ", p: ", p, ", graph_num: ", graph_num, ", init_num: ", init_num)

    params = ['python', 'single_1D.py',
              '--N', N,
              '--d', d,
              '--p', p,
              '--graph_num', graph_num,
              '--init_num', init_num,
              '--result_save_path', result_save_path]

    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    with open(os.path.join(result_save_path, "stdout-" + ib_ind + '-' + sample_num + ".txt"), "w+") as f_for_stdout:
        subprocess.run(params, stdout = f_for_stdout, stderr = subprocess.STDOUT)

N = 6
d = 4
num_random_graphs = 10
num_init_states = 1
p_noise_list = [0.001, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.00]

sweep_1D(N, d, num_random_graphs, num_init_states, p_noise_list)
