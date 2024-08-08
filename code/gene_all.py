
import os
import numpy as np
from network import QuantumNetwork
from utils import set_random_seed
import argparse

set_random_seed(0)

algorithm_names = ["Vanilla NB"]
algorithm_name=algorithm_names[0]

dis=['poisson', 'exponential', 'uniform', 'normal', 'pareto', 'lognormal', 'real']
noise_model_names = ["Depolar", "Dephase", "AmplitudeDamping", "BitFlip"]

cishu = 200
bounces = [1, 2, 3, 4]
sample_times = {}
for i in bounces:
    sample_times[i] = cishu

repeat=10

#for i0 in dis:
#for noise_model in noise_model_names:

parser = argparse.ArgumentParser(description='')
parser.add_argument('--distribution', type=str, default='poisson')
parser.add_argument('--noise', type=str, default='Depolar')
canshu = parser.parse_args()

i0=canshu.distribution
noise_model=canshu.noise

du=np.load('./gene_path/'+i0+'.npy')
if i0=='real':
    path_num=1500
else:
    path_num=100
path_list = list(range(1, path_num + 1))
fidelity_list=du[:path_num]
jieguo=np.zeros((repeat, path_num, 4, cishu))
for i1 in range(repeat):
    network = QuantumNetwork(path_num, fidelity_list, noise_model)
    for path in path_list:
        print(i0+' '*5+noise_model+' '*5+str(i1)+' '*5+str(path)+' '*5)
        raw_data = network.all_path(path, bounces, sample_times)
        jieguo[i1, path_list.index(path), 0]=np.array(raw_data[1])
        jieguo[i1, path_list.index(path), 1]=np.array(raw_data[2])
        jieguo[i1, path_list.index(path), 2]=np.array(raw_data[3])
        jieguo[i1, path_list.index(path), 3]=np.array(raw_data[4])

np.save('./qnet_data/'+i0+'_'+noise_model+'.npy', jieguo)

