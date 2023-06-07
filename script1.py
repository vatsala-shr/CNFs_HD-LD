import subprocess
shape = [1.0, 1.5, 2]
noise_iter = [1, 2, 4, 8, 16]
for j in range(len(noise_iter)):
    for i in range(len(shape)):
        command = f'python3 script_train.py --sup_ratio 1 --gpu_id 4 --crap_ratio 0.5 --shape {shape[i]} --noise_iter {noise_iter[j]} --noise True'
        subprocess.run(command, shell = True)