import numpy as np
import io, os, json
import concurrent.futures
import subprocess
import copy
import shutil
import pickle

from datetime import datetime
from datetime import date

def sweep_heis(N_list, p_list, theta_list, d_list, D_list):
    max_threads = 9

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
            seed_list = N + np.array(range(1))
            for seed in seed_list:
                for d in d_list: 
                    for p in p_list:
                        for D in D_list:
                            for theta in theta_list:
                                executor.submit(submit_simulation,
                                str(N), str(d), str(seed), str(p), str(D), str(theta), result_save_path)

def submit_simulation(N, d, seed, p, D, theta, result_save_path):
    '''
    Submit a simulation
    '''

    print('Submitting simulation with N: ', N, 'd: ', d, ', seed: ', seed,
          ", p: ", p, ", D: ", D, ", theta: ", theta)

    params = ['python', 'single_heis_1D.py',
              '--N', N,
              '--d', d,
              '--seed', seed,
              '--p', p,
              '--theta', theta,
              '--D', D,
              '--result_save_path', result_save_path]

    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    fname = "heis1D-N-" + str(N) + "-d-" + str(d) + "-seed-" + \
                seed + "-theta-" + f'{theta:.4f}' + \
                "-p-" + str(p) + "-D-" + str(D) + ".txt"
    with open(os.path.join(result_save_path, fname), "w+") as f_for_stdout:
        subprocess.run(params, stdout = f_for_stdout, stderr = subprocess.STDOUT)


N_list = [16]
p_list = [0.03, 0.1, 0.3]
d_list = [4, 6, 8, 10, 12, 20]
D_list = [2, 8, 32, 64, 128]
theta_list = [0.01, 0.1, 1.0]

sweep_heis(N_list, p_list, theta_list, d_list, D_list)