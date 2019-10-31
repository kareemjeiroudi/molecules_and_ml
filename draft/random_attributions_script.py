from modules.IntegratedGradients import integrated_gradients as IG

from keras.models import load_model

import pickle 

import time

import numpy as np
import pandas as pd


model = load_model('analysis_best_model/best_model_bayesian_optimization_3487.h5')
ig = IG(model)

with open('dumped_objects/bits_test.pckl', 'rb') as f:
    X_test, y_test = pickle.load(f)


_start_time = time.time()
def tic():
    global _start_time 
    _start_time = time.time()

def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))


## takes around 30 min
attributions_randoms = []
baselines_randoms = []
tic()
for i in range(50):
    attributions_random = {}
    baseline_random = np.random.randint(0, high=2, size=X_test.shape[1])
    baselines_randoms.append(baseline_random)
    for step in range(50, 101, 10):
        attributions_random[step] = np.array([ig.explain(inp, reference=baseline_random, num_steps=step) for inp in X_test])
    attributions_randoms.append(attributions_random)
    tac()
tac()


with open('dumped_objects/attributions_randoms.pckl', 'wb') as f:
    pickle.dump([attributions_randoms, baselines_randoms], f)