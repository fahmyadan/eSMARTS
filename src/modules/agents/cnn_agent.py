import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np 


class CNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CNNAgent, self).__init__()
        self.args = args

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], out_channels=32, kernel_size=8, stride=4 ),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        with torch.no_grad():
            #Initialise the zeros with shape (batch_size = 4-agents, channels, height, width )
            conv_out = self.conv_layers(torch.zeros(input_shape[-1], *input_shape[:-1])).view(4,-1).shape
            

        self.fc1 = nn.Linear(args.hidden_dim, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(conv_out[1], args.hidden_dim)
        else:
            self.rnn = nn.Linear(conv_out[0] * conv_out[1], args.hidden_dim)

        self.fc1_5 = nn.Linear(args.hidden_dim, args.out_dim)
        self.fc2 = nn.Linear(args.out_dim, args.n_actions)


    def init_hidden(self):
        #TODO: Investigate why hidden weights must be initialised like this 
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, input, hidden_state):
        # Process the state through the convolutional layers
        state = input.squeeze(0)

        try :
            conv_out = self.conv_layers(state)
        except RuntimeError as e:
            raise ValueError("Expected tensor of shape (4,3,112,112), but got tensor of shape {}".format(tuple(state.size()))) from e


        spatial_features = conv_out.view(4,-1)
        # flat_conv = conv_out.flatten()
        # x = F.relu(self.fc1(flat_conv))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(spatial_features, h_in)
        else:
            h = F.relu(self.rnn(conv_out.flatten()))
        
        l = self.fc1(h)
        f = self.fc1_5(l)
        q = self.fc2(f)
        return q, h
