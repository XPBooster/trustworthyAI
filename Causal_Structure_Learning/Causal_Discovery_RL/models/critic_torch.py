import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, config, is_train=True):
        super(Critic, self).__init__()

        self.num_nodes = config.num_nodes
        self.fcn1 = nn.Linear(config.d_model, config.d_model)
        self.fcn2 = nn.Linear(config.d_model, 1)

    def forward(self, encoder_output):
        # [Batch size, Sequence Length, Num_neurons] to [Batch size, Num_neurons]
        frame = torch.mean(encoder_output, dim=1)
        output = self.fcn1(frame)
        output = torch.relu(output)
        output = self.fcn2(output)
        return output.squeeze() # (batch,)