import os
import pickle

from smarts.sstudio import types as t
from smarts.sstudio.genscenario import gen_agent_missions, gen_social_agent_missions

scenario = os.path.dirname(os.path.realpath(__file__))

with open(os.environ["SOCIAL_AGENT_PATH"], "rb") as f:
    social_agent = pickle.load(f)

gen_social_agent_missions(
    scenario,
    social_agent_actor=social_agent,
    name=f"s-agent-{social_agent.name}",
    missions=[
        t.Mission(t.Route(begin=("E35-3", 2, 90), end=("E3-3s", 0, 30))),
    ],
)

gen_agent_missions(
    scenario,
    [
        t.Mission(t.Route(begin=("E3l-3", 1, 200), end=("E3-35", 1, 50))),
    ],
)
