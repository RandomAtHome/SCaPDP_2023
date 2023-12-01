# PYTHON 2.7
# mpirun -np 1 /usr/bin/python  <script_name>.py 

import os
import json
import subprocess

# initial params
n_tr = 5
prog_names = ["heat-3d-for", "heat-3d-task"]
compile_prefixes = ["gcc -fopenmp -std=c99 -O3",
                    "clang -fopenmp -std=c99 -O3",
                    "pgcc -mp -O3",
                    "icc -qopenmp -std=c99 -O3"]
data_sizes = ["MINI_DATASET",
              "SMALL_DATASET",
              "MEDIUM_DATASET",
              "LARGE_DATASET",
              "EXTRALARGE_DATASET"]
num_threads = [1, 2, 4, 8]

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

            # change data size
            change_code("{0}.c".format(prog_name), data_size)

            proc = subprocess.Popen(
                "{0} {1}.c heat-3d.h -D{2} -o {1}".format(compile_prefix, prog_name, data_size),
                # base args
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = proc.communicate()

            for num_thread in num_threads:
                data_dict[prog_name][compile_prefix.split()[0]][eval(data_size)][num_thread] = -1

                # update env
                base_env["OMP_NUM_THREADS"] = str(num_thread)

                full_time = 0
                for _ in range(n_tr):
                    proc = subprocess.Popen(
                        "./{0}".format(prog_name),
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
                print(
                    "RUN COMMAND: {0} {1}.c heat-3d.h -D{5} -o {1} with DATA SIZE: {2} and NUM_THREADS: {3} --- avgt: {4}".format(
                        compile_prefix, prog_name, eval(data_size), num_thread, full_time / n_tr, data_size))

with open("diff_data.json", "w") as fp:
    json.dump(data_dict, fp, indent=4)
