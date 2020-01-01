# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# +
#PCA over embeddings
#simple attention

# +
#consider training the embeddings

# +
class baseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_output_classes, window_size):
        super(baseLSTM, self).__init__()
        
        self.window_size=window_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_output_classes = num_output_classes

        self.fc6 = nn.Linear(input_dim, input_dim*5, bias = False)
        self.fc7 = nn.Linear(input_dim*5, input_dim*2, bias = False)
        self.fc8 = nn.Linear(input_dim*2, input_dim, bias = False)
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, bias = False)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim*2, bias = False)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim, bias = False)
        self.fc3 = nn.Linear(hidden_dim, int(hidden_dim/2), bias = False)
        self.fc4 = nn.Linear(int(hidden_dim/2), int(hidden_dim/5), bias = False)
        self.fc5 = nn.Linear(int(hidden_dim/5), int(hidden_dim/10), bias = False)
        self.fc9 = nn.Linear(int(hidden_dim/10), int(hidden_dim/20), bias = False)
        self.fc10 = nn.Linear(int(hidden_dim/20), num_output_classes, bias = False)
        
        self.dropout = nn.Dropout(p = 0.2)
        #self.softmax = nn.Softmax(dim = 1)

    def attention_net(self, all_hidden_states, next_hidden_state):

        attn_weights = torch.bmm(all_hidden_states.reshape(1, self.window_size, self.hidden_dim), next_hidden_state.reshape(1,self.hidden_dim,1))
        soft_attn_weights = F.softmax(attn_weights, dim = 1).reshape(1, 1, self.window_size)

        new_hidden_state = torch.bmm(soft_attn_weights, torch.transpose(all_hidden_states,0,1))
        #https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py
        return new_hidden_state

    def forward(self, input_, hidden_state, cell_state):
        
        fc_on_row = self.fc6(input_)
        fc_on_row = self.fc7(fc_on_row)
        input_ = self.fc8(fc_on_row)
        
        all_hidden_states, (next_hidden_state, next_cell_state) = self.lstm(input_, (hidden_state, cell_state))
        # note - all_hidden_states[-1] = next_hidden_state
        
        next_hidden_state = self.softmax(next_cell_state, dim = 1)
        
        next_hidden_state = self.attention_net(all_hidden_states, next_hidden_state)
        
        #print(next_hidden_state)
        
        fc_on_row = self.fc1(next_hidden_state)
        fc_on_row = self.fc2(fc_on_row)
        fc_on_row = self.fc3(fc_on_row)
        fc_on_row = self.fc4(fc_on_row)
        fc_on_row = self.fc5(fc_on_row)
        fc_on_row = self.fc9(fc_on_row)
        out = self.fc10(fc_on_row)
        
        out = out.view(1, out.shape[2])
        #out = nn.ReLU()(out)
        
        return out, (next_hidden_state, next_cell_state)


