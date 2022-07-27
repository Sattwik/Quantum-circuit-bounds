import numpy as np
import io, os, json
import concurrent.futures
import subprocess
import copy
import shutil
import pickle

from datetime import datetime
from datetime import date

def sweep_fermion(N_list, p_list):
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
        for N in N_list:
            seed_list = N + np.array(range(5))
            for seed in seed_list:
                for p in p_list:
                    for k_dual in [1, N]:
                        executor.submit(submit_simulation,
                        str(N), str(seed), str(p), str(k_dual), result_save_path)

def submit_simulation(N, seed, p, k_dual, result_save_path):
    '''
    Submit a simulation
    '''

    print('Submitting simulation with N: ', N, ', seed: ', seed,
          ", p: ", p, ", k_dual: ", k_dual)

    params = ['python', 'single_fermion.py',
              '--N', N,
              '--seed', seed,
              '--p', p,
              '--k_dual', k_dual,
              '--result_save_path', result_save_path]

    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    fname = "fermion1D-N-" + str(N) + "-seed-" + seed + "-p-" + str(p) + "-kdual-" + \
                     str(k_dual) + ".txt"
    with open(os.path.join(result_save_path, fname), "w+") as f_for_stdout:
        subprocess.run(params, stdout = f_for_stdout, stderr = subprocess.STDOUT)

N_list = [10,15,20,25,30]
p_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
sweep_fermion(N_list, p_list)
