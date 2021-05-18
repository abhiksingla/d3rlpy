from sklearn.model_selection import train_test_split
import d3rlpy
from d3rlpy.datasets import get_pybullet
from d3rlpy.algos import CQL
from d3rlpy.ope import FQE
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import soft_opc_scorer

dataset, env = get_pybullet('hopper-bullet-mixed-v0')

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

# train algorithm
cql = CQL()
cql.fit(train_episodes,
        eval_episodes=test_episodes,
        n_epochs=10,
        scorers={
            'environment': evaluate_on_environment(env),
            'init_value': initial_state_value_estimation_scorer,
            'soft_opc': soft_opc_scorer(600)
        })

# or load the trained model
cql = CQL.from_json('d3rlpy_logs/CQL_20210517040138/params.json')
cql.load_model('d3rlpy_logs/CQL_20210517040138/model_10.pt')

# evaluate the trained policy
fqe = FQE(algo=cql,
          q_func_factory='qr',
          learning_rate=1e-4,
          use_gpu=False,
          encoder_params={'hidden_units': [512, 256]})
fqe.fit(dataset.episodes,
        n_epochs=5,
        eval_episodes=dataset.episodes,
        scorers={
            'init_value': initial_state_value_estimation_scorer,
            'soft_opc': soft_opc_scorer(600)
        })
