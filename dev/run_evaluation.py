from sklearn.model_selection import train_test_split
import d3rlpy
from d3rlpy.datasets import get_pybullet
from d3rlpy.algos import CQL, SAC
from d3rlpy.ope import FQE
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer, average_value_estimation_scorer
from d3rlpy.metrics.scorer import soft_opc_scorer
import numpy as np
from collections import defaultdict
import time
import json

dataset_name = 'hopper-bullet-mixed-v0'
dataset, env = get_pybullet(dataset_name)

scorers={
            'init_value': initial_state_value_estimation_scorer,
            'average_value_estimation': average_value_estimation_scorer
        }

path = "d3rlpy_logs/"
algo_log = "CQL_20210517232215/"
agent = CQL.from_json(path + algo_log + 'params.json')

# evaluate the trained policy
fqe = FQE(algo=agent)

results = defaultdict(list)
prev_time = time.time()
for iter in range(1,31):
    model = f"model_{iter}.pt"
    agent.load_model(path + algo_log + model)
    fqe.fit(dataset=dataset.episodes,
            n_epochs=30,
            eval_episodes=dataset.episodes,
            scorers=scorers,
            tensorboard_dir="d3rlpy_logs/",
            save_interval=10)
    assert len(fqe._eval_results["init_value"])==30

    curr_time = time.time()
    results["iter"].append(iter)
    results['init_value'].append(fqe._eval_results["init_value"][-1])
    results['average_value_estimation'].append(fqe._eval_results["average_value_estimation"][-1])
    results["time"].append(curr_time - prev_time)
    prev_time = curr_time
    if iter%5==0:
        with open('fqe_results.json', 'w') as f:
            json.dump(results, f)

results['mean_time'] = sum(results['time'])/len(results['time'])

with open('fqe_results.json', 'w') as f:
    json.dump(results, f)

