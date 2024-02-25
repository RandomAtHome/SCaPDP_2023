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
            print("changed in", path, "from", sep_code_file[i], "to", "#define N {0}".format(data))
            sep_code_file[i] = "#define N {0}".format(data)
    with open(path, "w") as fp:
        for line in sep_code_file:
            fp.write("{0}\n".format(line))


# initial params
n_tr = 5
prog_names = ["var11_for", "var11_task"]
compile_prefixes = [
                    "icc -qopenmp -std=c99 -O3",
                    # "gcc -fopenmp -std=c99 -O3",
                    # "clang -fopenmp -std=c99 -O3",
                    # "pgcc -mp -O3",
                    ]
# data_sizes = ["((1<<10)+2)", "((1<<11)+2)", "((1<<12)+2)", "((1<<13)+2)"]
data_sizes = ["1026", "2050", "4098", "8194"]
num_threads = [1, 2, 4, 8, 16]

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

            for num_thread in num_threads:
                data_dict[prog_name][compile_prefix.split()[0]][eval(data_size)][num_thread] = -1

                # update env
                base_env["OMP_NUM_THREADS"] = str(num_thread)

                full_time = 0
                for _ in range(n_tr):
                    proc = subprocess.Popen(
                        "./{0}_{1}".format(prog_name, data_size),
                        env=base_env,
                        # base args
                        shell=True,
                        executable="/bin/bash",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    stdout, stderr = proc.communicate()
                    full_time += float(stdout)

                data_dict[prog_name][compile_prefix.split()[0]][eval(data_size)][num_thread] = full_time / n_tr
                print("RUN COMMAND: {0} {1}.c -o {1} with DATA SIZE: {2} and NUM_THREADS: {3} --- avgt: {4}".format(
                    compile_prefix, prog_name, eval(data_size), num_thread, full_time / n_tr))

with open("diff_data_icc.json", "w") as fp:
    json.dump(data_dict, fp, indent=4)
