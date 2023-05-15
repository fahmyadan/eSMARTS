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
            nn.Conv2d(input_shape[1], out_channels=16, kernel_size=8, stride=4 ),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )

        
        with torch.no_grad():
            #Initialise the zeros with shape (batch_size =4, channels, height, width )
            # conv_out = self.conv_layers(torch.zeros(input_shape)).view(args.n_agents,-1).shape
            conv_out = self.conv_layers(torch.zeros((1,) + input_shape[1:])).view(1,-1)
            

        self.fc1 = nn.Linear(conv_out.size(1), args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.fc1_5 = nn.Linear(args.hidden_dim, args.out_dim)
        self.fc2 = nn.Linear(args.out_dim, args.n_actions)


    def init_hidden(self):
        #TODO: Investigate why hidden weights must be initialised like this 
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, input, hidden_state):
        batch_size = input.size(0)
        num_agents = input.size(1)
        # Process the state through the convolutional layers
        # state = input.squeeze(0)
        state = input.view(-1, input.size(2), input.size(3), input.size(4))
        

        conv_out = self.conv_layers(state)

        # spatial_features = conv_out.view(4,-1)
        spatial_features = conv_out.view(batch_size, num_agents, -1)
        # flat_conv = conv_out.flatten()
        # x = F.relu(self.fc1(flat_conv))

        mlp_out =  self.fc1(spatial_features).view(batch_size *num_agents, -1)


        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(mlp_out, h_in)
        else:
            h = F.relu(self.rnn(conv_out.flatten()))
        
    
        f = self.fc1_5(h)
        q = self.fc2(f)
        return q, h
