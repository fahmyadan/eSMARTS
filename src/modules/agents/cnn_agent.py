import torch 
import torch.nn as nn
import torch.nn.functional as F

class CNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CNNAgent, self).__init__()
        self.args = args

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[-1], out_channels=32, kernel_size=8, stride=4 ),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        with torch.no_grad():
            conv_out = self.conv_layers(torch.zeros(1, *input_shape)).flatten().shape[0]

        self.fc1 = nn.Linear(conv_out, args.hidden_dim)
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
        # Process the state through the convolutional layers
        conv_out = self.conv_layers(state)
        conv_out = conv_out.flatten(start_dim=1)


        x = F.relu(self.fc1(conv_out))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        f = self.fc1_5(h)
        q = self.fc2(f)
        return q, h
