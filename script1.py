import subprocess
shape = [0.75]
noise_iter = [4]
for j in range(len(noise_iter)):
    for i in range(len(shape)):
        command = f'python3 train.py --sup_ratio 1 --gpu_id 0 --crap_ratio 0.5 --shape {shape[i]} --noise_iter {noise_iter[j]} --noise True'
        subprocess.run(command, shell = True)