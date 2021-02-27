from functools import partial
from envs.multiagentenv import MultiAgentEnv
from envs.macad_env import MacadEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["macad"] = partial(env_fn, env=MacadEnv)
