
import os
import math
import time
import warnings
import numpy as np
import random as rd
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")

bounces=[1,2,3,4]

def EXP(x, p, A):
    return A * p**(2 * x)

def REGRESSION(bounces, mean_bm):
    try:
        popt_AB, pcov_AB = curve_fit(EXP, bounces, mean_bm, p0=[0.9, 0.5], maxfev=1000)
        xx = popt_AB[0]
        if popt_AB[0]>1:
            xx = 1.0
        elif popt_AB[0]<0:
            xx = 0.1
        return [xx, popt_AB[1]]
    except RuntimeError as e:
        return [0.1, 0.1]

def data_processing(raw_data):
    mean_values = raw_data.mean(axis=1)
    p, _ = REGRESSION(bounces, mean_values)
    return p

def naive_alg(juzhen):
    fidelity = {}
    cost = 0
    for path in range(juzhen.shape[0]):
        p = data_processing(juzhen[path])
        fidelity[path] = p + (1 - p) / 2  # Convert the estimated depolarizing parameter `p` into fidelity
        cost += 10*200
    best_path = max(fidelity, key=fidelity.get)
    best_path_fidelity = fidelity[best_path]
    return best_path, cost, best_path_fidelity

def succ_elim(juzhen):
    L = juzhen.shape[0]
    active_set = list(range(juzhen.shape[0]))
    C = 0.15; N = 4; cost = 0; delta = 0.1
    mean = {path: 0 for path in range(juzhen.shape[0])}
    n = {path: 0 for path in range(juzhen.shape[0])}
    t = 0
    while len(active_set) > 1:
        t += 1
        ucb = {}
        lcb = {}
        for path in active_set:
            #print(active_set)
            #print(juzhen[path][:, N*(t-1):N*t])
            p = data_processing(juzhen[path][:, N*(t-1):N*t])
            cost += 10*N
            mean[path] = (mean[path] * n[path] + p) / (n[path] + 1)
            n[path] += 1
            r = C * math.sqrt(math.log(4 * L * t * t / delta) / (2 * t))
            # print(f"r={r}, {math.log(4 * L * t * t / delta)}")
            ucb[path] = mean[path] + r
            lcb[path] = mean[path] - r
        new_active_set = []
        for path1 in active_set:
            ok = True
            for path2 in active_set:
                if path1 != path2 and ucb[path1] < lcb[path2]:
                    ok = False
                    break
            if ok:
                new_active_set.append(path1)
        active_set = new_active_set
        if N*t>=juzhen.shape[2]:
            break
    estimated_fidelity = {}
    for path in range(juzhen.shape[0]):
        p = mean[path]
        estimated_fidelity[path] = p + (1 - p) / 2
    best_path = max(estimated_fidelity, key=estimated_fidelity.get)
    best_path_fidelity = estimated_fidelity[best_path]
    return best_path, cost, best_path_fidelity

def linkselfie(juzhen):
    candidate_set = range(juzhen.shape[0])
    s = 0; C = 0.01; delta = 0.1; cost = 0; estimated_fidelities = {}
    wei=0
    while len(candidate_set) > 1:
        s += 1
        Ns = math.ceil(C * 2**(2 * s) * math.log2(2**s * len(candidate_set) / delta))
        if Ns < 4:
            Ns = 4
        p_s = {}
        for path in candidate_set:
            p = data_processing(juzhen[path][:, wei:Ns+wei])
            estimated_fidelities[path] = p + (1 - p) / 2
            p_s[path] = p
            cost += 10*Ns
        p_max = max(p_s.values())
        new_candidate_set = []
        for path in candidate_set:
            if p_s[path] + 2**(-s) > p_max - 2**(-s):
                new_candidate_set.append(path)
        candidate_set = new_candidate_set
        wei+=Ns
        if wei>=juzhen.shape[2]:
            break
    best_path = max(estimated_fidelities, key=estimated_fidelities.get)
    best_path_fidelity = estimated_fidelities[best_path]
    return best_path, cost, best_path_fidelity

def qbgp(juzhen):
    L = juzhen.shape[0]
    active_set = list(range(juzhen.shape[0]))
    C = 0.15; N = 4; cost = 0; delta = 0.05
    mean = {path: 0 for path in range(juzhen.shape[0])}
    n = {path: 0 for path in range(juzhen.shape[0])}
    t = 0
    while len(active_set) > 1:
        t += 1
        ucb = {}
        lcb = {}
        for path in active_set:
            #print(active_set)
            #print(juzhen[path][:, N*(t-1):N*t])
            p = data_processing(juzhen[path][:, N*(t-1):N*t])
            cost += 10*N
            mean[path] = (mean[path] * n[path] + p) / (n[path] + 1)
            n[path] += 1
            r = C * math.sqrt(math.log(2 * L * t * t / delta) / (2 * t))
            # print(f"r={r}, {math.log(4 * L * t * t / delta)}")
            ucb[path] = mean[path] + r
            lcb[path] = mean[path] - r
        new_active_set = []
        for path1 in active_set:
            ok = True
            for path2 in active_set:
                if path1 != path2 and ucb[path1] < lcb[path2]:
                    ok = False
                    break
            if ok:
                new_active_set.append(path1)
        active_set = new_active_set
        if N*t>=juzhen.shape[2]:
            break
    estimated_fidelity = {}
    for path in range(juzhen.shape[0]):
        p = mean[path]
        estimated_fidelity[path] = p + (1 - p) / 2
    best_path = max(estimated_fidelity, key=estimated_fidelity.get)
    best_path_fidelity = estimated_fidelity[best_path]
    return best_path, cost, best_path_fidelity

def qlsel(juzhen):
    #print('xxxx')
    candidate_set = range(juzhen.shape[0])
    t_a=np.ones(juzhen.shape[0]); t_b=np.ones(juzhen.shape[0])
    s = 0; cost = 0; estimated_fidelities = {}
    fed_all_path=np.zeros((juzhen.shape[0], juzhen.shape[2]))
    jilu=time.time()
    for i1 in range(juzhen.shape[0]):
        for i2 in range(juzhen.shape[2]):
            fed_all_path[i1,i2] = data_processing(juzhen[i1][:, i2:i2+1])
    #print(time.time()-jilu)
    jilu=time.time()
    wei=np.zeros((juzhen.shape[0])).astype(int)
    while len(candidate_set) > 1:
        #print(time.time()-jilu)
        jilu=time.time()
        s += 1
        p_s = {}
        for path in candidate_set:
            yong = fed_all_path[path][:wei[path]].copy()
            if wei[path]==0:
                Ns = 4
            else:
                Ns = int(4+10*max(0, (np.std(yong)/np.mean(yong))))
            wei[path]+=Ns
            #print(wei[path])
            lin = fed_all_path[path, :min(wei[path], juzhen.shape[2])].copy()#
            #print(lin)
            lowlin = np.percentile(lin, 25)
            uplin = np.percentile(lin, 75)
            IQR = uplin - lowlin
            if IQR <= 0:
                IQR = 0.1
            #print(path, lin)
            #print(uplin, lowlin, IQR)
            p_s[path] = np.random.beta(uplin, IQR, size=10000).mean() #uplin
            cost += 10*Ns
        p_max = max(p_s.values())
        new_candidate_set = []
        for path in candidate_set:
            if p_s[path] + 2**(-s) > p_max - 2**(-s):
                new_candidate_set.append(path)
        candidate_set = new_candidate_set
        if wei.max()>=juzhen.shape[2]:
            break
    for i1 in range(juzhen.shape[0]):
        estimated_fidelities[i1] = fed_all_path[i1][:wei[i1]].mean()
    best_path = max(estimated_fidelities, key=estimated_fidelities.get)
    best_path_fidelity = estimated_fidelities[best_path]
    return best_path, cost, best_path_fidelity


dis=['real', 'poisson', 'exponential', 'uniform', 'normal', 'pareto', 'lognormal', ] #, 
noise_model_names = ["Depolar", "Dephase", "AmplitudeDamping", "BitFlip"] #

for i0 in dis:
    du=np.load('./gene_path/'+i0+'.npy')
    if i0=='real':
        path_num=1500
    else:
        path_num=100
    fidelity_list=du[:path_num]
    for noise_model in noise_model_names:
        jieguo=np.load('./qnet_data/'+i0+'_'+noise_model+'.npy')
        all_res=np.zeros((5, jieguo.shape[0], 4))
        print(i0+'==='+noise_model)
        for i1 in range(jieguo.shape[0]):
            best_p, cost_sum, best_p_fed = naive_alg(jieguo[i1])
            print('naive_alg', fidelity_list[best_p], '-'*5, fidelity_list.max(), '-'*5, best_p_fed, '-'*5, cost_sum)
            all_res[0, i1]=[fidelity_list[best_p], fidelity_list.max(), best_p_fed, cost_sum]
            best_p, cost_sum, best_p_fed = succ_elim(jieguo[i1])
            print('succ_elim', fidelity_list[best_p], '-'*5, fidelity_list.max(), '-'*5, best_p_fed, '-'*5, cost_sum)
            all_res[1, i1]=[fidelity_list[best_p], fidelity_list.max(), best_p_fed, cost_sum]
            best_p, cost_sum, best_p_fed = linkselfie(jieguo[i1])
            print('linkselfie', fidelity_list[best_p], '-'*5, fidelity_list.max(), '-'*5, best_p_fed, '-'*5, cost_sum)
            all_res[2, i1]=[fidelity_list[best_p], fidelity_list.max(), best_p_fed, cost_sum]
            best_p, cost_sum, best_p_fed = qbgp(jieguo[i1])
            print('qbgp', fidelity_list[best_p], '-'*5, fidelity_list.max(), '-'*5, best_p_fed, '-'*5, cost_sum)
            all_res[3, i1]=[fidelity_list[best_p], fidelity_list.max(), best_p_fed, cost_sum]
            best_p, cost_sum, best_p_fed = qlsel(jieguo[i1])
            print('qlsel', fidelity_list[best_p], '-'*5, fidelity_list.max(), '-'*5, best_p_fed, '-'*5, cost_sum)
            all_res[4, i1]=[fidelity_list[best_p], fidelity_list.max(), best_p_fed, cost_sum]
        np.save('./qnet_res/'+i0+'_'+noise_model+'.npy', all_res)
