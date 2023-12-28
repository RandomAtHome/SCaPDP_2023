# PYTHON 2.7
# mpirun -np 1 /usr/bin/python  <script_name>.py 

import os
import json
import subprocess

# initial params
n_tr = 5
prog_names = ["heat-3d-acc"]
compile_prefixes = ["pgcc -fopenmp -std=c99 -fast"]
modes = ["tesla", "multicore", "host"]
data_sizes = ["MINI_DATASET",
              "SMALL_DATASET",
              "MEDIUM_DATASET",
              "LARGE_DATASET",
              "EXTRALARGE_DATASET"]

# store our pc env
base_env = os.environ.copy()

# data storage
data_dict = {}

# interate over prog names
for prog_name in prog_names:
    data_dict[prog_name] = {}
    # iterate over compilers
    for compile_prefix in compile_prefixes:
        data_dict[prog_name][compile_prefix.split()[0]] = {}
        # iterate over opt
        for data_size in data_sizes:
            data_dict[prog_name][compile_prefix.split()[0]][data_size] = {}

            for mode in modes:
                new_comp_prefix = compile_prefix + " -ta={}".format(mode)
                if not os.path.exists("{0}_{1}_{2}".format(new_comp_prefix, prog_name, new_comp_prefix.split()[0], data_size)):

                proc = subprocess.Popen(
                    "{0} {1}.c -D{2} -o {1}".format(compile_prefix, prog_name, data_size),
                    # base args
                    shell=True,
                    executable="/bin/bash",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

            
                data_dict[prog_name][compile_prefix.split()[0]][data_size][mode] = -1

                full_time = 0
                for _ in range(n_tr):
                    proc = subprocess.Popen(
                        "./{0}_{1}_{2}".format(prog_name, new_comp_prefix.split()[0], data_size),
                        env=base_env,
                        # base args
                        # shell=True,
                        # executable="/bin/bash", 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                    )
                    stdout, stderr = proc.communicate()
                    print(stdout)
                    print(stderr)
                    try:
                        full_time += float(stdout)
                    except:
                        full_time += 0

                data_dict[prog_name][compile_prefix.split()[0]][data_size][mode] = full_time / n_tr
                print("RUN COMMAND: {0} {1}.c -o {1}_{2}_{3} with DATA SIZE: {3} and MODE: {4} --- avgt: {5}".format(new_comp_prefix, prog_name, new_comp_prefix.split()[0], data_size, mode, full_time / n_tr))

with open("diff_opt_data.json", "w") as fp:
    json.dump(data_dict, fp, indent=4)