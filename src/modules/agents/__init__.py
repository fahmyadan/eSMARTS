REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn1_agent import RNN1Agent
from .cnn_agent import CNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY['rnn1'] = RNN1Agent
REGISTRY['cnn'] = CNNAgent