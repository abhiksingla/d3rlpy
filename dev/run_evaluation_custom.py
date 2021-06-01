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
from d3rlpy.models.encoders import VectorEncoderFactory

from d3rlpy.models.encoders import EncoderFactory
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)

# your own neural network
# class CustomEncoder(nn.Module):
#     def __init__(self, obsevation_shape, feature_size):
#         self.feature_size = feature_size
#         self.fc1 = nn.Linear(observation_shape[0], feature_size)
#
#     def forward(self, x):
#         return self.fc1(x)
#
#     # THIS IS IMPORTANT!
#     def get_feature_size(self):
#         return self.feature_size

class CustomEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size):
        super(CustomEncoder, self).__init__()
        self.feature_size = feature_size
        self.fc1 = nn.Linear(observation_shape[0], feature_size)

    def forward(self, x):
        return self.fc1(x)

    # THIS IS IMPORTANT!
    def get_feature_size(self):
        return self.feature_size

class CustomEncoderWithAction(nn.Module):
    def __init__(self, observation_shape, action_size, feature_size):
        super(CustomEncoderWithAction, self).__init__()
        self.feature_size = feature_size
        self.action_size = action_size
        self.fc1 = nn.Linear(observation_shape[0] + action_size, feature_size)

    def forward(self, x, action): # action is also given
        h = torch.cat([x, action], dim=1)
        return self.fc1(h)

    def get_feature_size(self):
        return self.feature_size

class CustomEncoderFactory(EncoderFactory):
    TYPE = 'custom' # this is necessary

    def __init__(self, feature_size):
        self.feature_size = feature_size

    def create(self, observation_shape):
        return CustomEncoder(observation_shape, self.feature_size)

    def create_with_action(
        self,
        observation_shape,
        action_size: int,
        discrete_action: bool = False,
    ):
        return CustomEncoderWithAction(observation_shape, action_size, self.feature_size)

    def get_params(self, deep=False):
        return {'feature_size': self.feature_size}



def evaluate(algo_log, algo_name, dataset, mode='nn'):

    path = "d3rlpy_logs/"
    algo_log = algo_log
    if algo_name =='sac':
        agent = SAC.from_json(path + algo_log + 'params.json')
    else:
        agent = CQL.from_json(path + algo_log + 'params.json')

    if mode!='nn':
        encoder_factory = CustomEncoderFactory(feature_size=128)
        fqe = FQE(algo=agent, encoder_factory=encoder_factory)
    else:
        fqe = FQE(algo=agent)

    logger.debug(f"Agent: {agent}")
    logger.debug(f"FQE: {fqe}")

    results = defaultdict(list)
    prev_time = time.perf_counter()
    for iter in range(1,31):
        model = f"model_{iter}.pt"
        agent.load_model(path + algo_log + model)
        fqe.fit(dataset=dataset,
                n_epochs=30,
                eval_episodes=dataset,
                scorers=scorers,
                save_interval=30)
        assert len(fqe._eval_results["init_value"])==30

        curr_time = time.perf_counter()
        results["iter"].append(iter)
        results['init_value'].append(fqe._eval_results["init_value"][-1])
        results['average_value_estimation'].append(fqe._eval_results["average_value_estimation"][-1])
        results["time"].append(curr_time - prev_time)
        prev_time = curr_time
        if iter%5==0:
            with open(f'fqe_results_{algo_name}_{mode}.json', 'w') as f:
                json.dump(results, f)

    results['mean_time'] = sum(results['time'])/len(results['time'])

    with open(f"fqe_results_{algo_name}_{mode}.json", 'w') as f:
        json.dump(results, f)

scorers={
            'init_value': initial_state_value_estimation_scorer,
            'average_value_estimation': average_value_estimation_scorer
        }

def main():
    dataset_name = 'hopper-bullet-random-v0'
    dataset, env = get_pybullet(dataset_name)
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.05, random_state=1)

    evaluate(algo_log="CQL_20210521121424/", algo_name="cql", dataset=test_episodes)
    evaluate(algo_log="CQL_20210521121424/", algo_name="cql", dataset=test_episodes, mode='linear')
    # evaluate(algo_log="SAC_20210517231728/", algo_name="sac")

if __name__ == "__main__":
    main()
