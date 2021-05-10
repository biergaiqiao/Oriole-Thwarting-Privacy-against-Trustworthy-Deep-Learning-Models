import sys

sys.path.append("/home/shansixioing/tools")
from gen_utils import master_run
import random
import numpy as np
import glob

seed = 12342
random.seed(seed)
np.random.seed(seed)
import time


def main():
    # gpu_ls = ['babygroot0', 'babygroot1', 'babygroot3', 'groot0', 'groot1', 'groot2', 'groot3', 'nebula0',
    #           'nebula1', 'nebula2']
    # gpu_ls = ['george0', 'george1', 'george2', 'george3', 'fred0', 'fred1', 'fred2', 'nebula0', 'nebula1',
    #           'nebula2']
    gpu_ls = {
        # 'george0': 3,
        # 'george1': 2,
        'george2': 1,
        'george3': 1,
        # 'fred0': 2,
        # 'fred1': 2,
        # 'fred2': 1,
        # 'fred3': 1,
        # 'nebula0': 3,
        # 'nebula1': 3,
        # 'nebula2': 3,
        # # 'babygroot0': 2,
        # 'babygroot1': 2,
        # 'babygroot2': 2,
        # 'babygroot3': 2,
        # 'groot0': 2,
        # 'groot1': 2,
        # 'groot2': 2,
        # 'groot3': 2,
    }

    all_queries_to_run = []

    exp_names = []
    for directory in glob.glob("/home/shansixioing/data/fawkes_test_small2/*/"):
        exp_names.append(directory)
    # , 'high'
    print(exp_names)
    time.sleep(2)
    for mode in ['high']:
        for exp_name in exp_names:
            arg_string = "python3 protection.py -d {} -m {} --batch-size 20 -g {} --debug".format(
                exp_name, mode, "GPUID"
            )
            print(arg_string)
            args = arg_string.split(" ")
            args = [str(x) for x in args]
            all_queries_to_run.append(args)

    master_run(all_queries_to_run, gpu_ls, max_num=None, rest=1)

    print("Completed")


if __name__ == '__main__':
    main()
