# PYTHON 2.7
# mpirun -np 1 /usr/bin/python  <script_name>.py 

import os
import json
import subprocess

# initial params
n_tr = 5
prog_names = ["var11_for", "var11_task"]
compile_prefixes = [
    # "gcc -fopenmp -std=c99",
                    # "clang -fopenmp -std=c99",
                    # "pgcc -mp",
                    "icc -qopenmp -std=c99",
                    ]
compiler_optimizers = ["O0", "O1", "O2", "O3"]

# store our pc env
base_env = os.environ.copy()
base_env["OMP_NUM_THREADS"] = "8"

# data storage
data_dict = {}

# interate over prog names
for prog_name in prog_names:
    data_dict[prog_name] = {}
    # iterate over compilers
    for compile_prefix in compile_prefixes:
        data_dict[prog_name][compile_prefix.split()[0]] = {}
        # iterate over opt
        for compiler_optimizer in compiler_optimizers:
            data_dict[prog_name][compile_prefix.split()[0]][compiler_optimizer] = -1.0

            full_time = 0
            for _ in range(n_tr):
                proc = subprocess.Popen(
                    "./{0}_{1}".format(prog_name, compiler_optimizer),
                    env=base_env,
                    # base args
                    shell=True,
                    executable="/bin/bash",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, stderr = proc.communicate()
                full_time += float(stdout)

            data_dict[prog_name][compile_prefix.split()[0]][compiler_optimizer] = full_time / n_tr
            print("RUN COMMAND: {0} -{1} {2}.c -o {2}_{1} --- avgt: {3}".format(compile_prefix, compiler_optimizer,
                                                                                prog_name, full_time / n_tr))

with open("diff_compiler_opt_icc.json", "w") as fp:
    json.dump(data_dict, fp, indent=4)
