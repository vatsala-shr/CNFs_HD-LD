import subprocess
supervision = [0.5, 1.0]
# supervision = [0.05, 0.1]
# supervision = [0.1, 0.2, 0.5, 1.0]
# supervision = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
shape = [0.5, 0.75, 1.0, 1.5, 2]
for i in range(len(shape)):
    command = f'python3 script_train.py --sup_ratio 1 --gpu_id 4 --crap_ratio 0.5 --shape {shape[i]}'
    subprocess.run(command, shell = True)