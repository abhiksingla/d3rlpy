from sklearn.model_selection import train_test_split
import d3rlpy
from d3rlpy.datasets import get_pybullet
from d3rlpy.algos import CQL, SAC
from d3rlpy.ope import FQE
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import soft_opc_scorer

dataset_name = 'hopper-bullet-mixed-v0'
dataset, env = get_pybullet(dataset_name)

# train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

# train algorithm
scorers={
            'environment': evaluate_on_environment(env),
            'init_value': initial_state_value_estimation_scorer,
        }

algo = CQL()
# algo = SAC()
algo.fit(dataset=dataset.episodes,
        eval_episodes=dataset.episodes,
        n_epochs=30,
        scorers=scorers,
        tensorboard_dir="d3rlpy_logs/")
