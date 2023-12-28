# PYTHON 2.7
# mpirun -np 1 /usr/bin/python  <script_name>.py 

import os
import json
import subprocess

# chenge sting of code with data size definition
def change_code(path, data):
    with open(path, "r") as fp:
        code_file = fp.read()
    sep_code_file = code_file.split("\n")
    for i in range(len(sep_code_file)):
        if "#define N" in sep_code_file[i]:
            sep_code_file[i] = "#define N {0}".format(data)
    with open(path, "w") as fp:
        for line in sep_code_file:
            fp.write("{0}\n".format(line))

# initial params
n_tr = 5
prog_names = ["var11_acc"]
compile_prefixes = ["pgcc -fopenmp -std=c99 -fast"]
data_sizes = ["((1<<10)+2)", "((1<<11)+2)", "((1<<12)+2)", "((1<<13)+2)"]
modes = ["nvidia", "multicore", "host"]

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
            data_dict[prog_name][compile_prefix.split()[0]][eval(data_size)] = {}

            for mode in modes:
                new_comp_prefix = compile_prefix + " -ta={}".format(mode)
                if not os.path.exists("{0}_{1}_{2}".format(new_comp_prefix, prog_name, new_comp_prefix.split()[0], eval(data_size))):
                    # change data size
                    change_code("{0}.c".format(prog_name), data_size)

                    proc = subprocess.Popen(
                        "{0} {1}.c -o {1}_{2}_{3}".format(new_comp_prefix, prog_name, new_comp_prefix.split()[0], eval(data_size)), 
                        # base args
                        shell=True,
                        executable="/bin/bash", 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                    )
                    stdout, stderr = proc.communicate()

            
                data_dict[prog_name][compile_prefix.split()[0]][eval(data_size)][mode] = -1

                full_time = 0
                for _ in range(n_tr):
                    proc = subprocess.Popen(
                        "./{0}_{1}_{2}".format(prog_name, new_comp_prefix.split()[0], eval(data_size)), 
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

                data_dict[prog_name][compile_prefix.split()[0]][eval(data_size)][mode] = full_time / n_tr
                print("RUN COMMAND: {0} {1}.c -o {1}_{2}_{3} with DATA SIZE: {3} and MODE: {4} --- avgt: {5}".format(new_comp_prefix, prog_name, new_comp_prefix.split()[0], eval(data_size), mode, full_time / n_tr))

with open("diff_opt_data.json", "w") as fp:
    json.dump(data_dict, fp, indent=4)