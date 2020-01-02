import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from grid_search import perform_gridSearch
from customLSTM import baseLSTM
from trainer import train
from tester import test

import os
import pandas as po

input_dim = 23
num_output_classes = 5
hidden_dim = 2000
num_epochs = 5
window_size = 5
batch_size = 1 #number of tickers to be passed at the same time
dropout_prob = 0.2

ticker = 'CLc1'
train_from = '2009-01-01' # >= applied
train_until = '2009-02-01' # <= applied
test_from = '2009-02-02' # >= applied
test_until = '2009-02-10' # <= applied

retrain_while_testing = True
retrain_after = 23

df = po.read_csv('merged_data_without_embeddings/' + ticker + 'merged.csv')
df = df[df['No. Trades'] != 0].reset_index(drop = True)
val_cts = df['class'].value_counts().to_numpy()
del df 

val_cts = val_cts/sum(val_cts)
weights = torch.tensor(val_cts, dtype = torch.float)
weights

model = baseLSTM(input_dim, hidden_dim, num_output_classes, window_size)
loss_function = nn.CrossEntropyLoss(ignore_index = 5, weight = weights)
optimizer = optim.Adam(model.parameters(), lr=0.1)

best_scores = perform_gridSearch(ticker, window_size, train_from, train_until, test_from, test_until, model, loss_function, optimizer, num_epochs, input_dim, num_output_classes, hidden_dim, retrain_while_testing, retrain_after, dropout_prob) 

model, (hidden_state, cell_state) = train(ticker, window_size, train_from, train_until, model, loss_function, optimizer, num_epochs, input_dim, num_output_classes, hidden_dim, dropout_prob)

torch.save(model.state_dict(), 'saved_models/'+ticker+'_'+str(hidden_dim)+'_' + train_from + '_' + train_until + '_'+str(num_epochs) + '_no_embeddings_warm_start_attention')

predictions_df, model = test(ticker, window_size, test_from, test_until, retrain_while_testing, retrain_after, model, hidden_state, cell_state, loss_function, optimizer, num_epochs, input_dim, num_output_classes, hidden_dim, dropout_prob)

torch.save(model.state_dict(), 'saved_models/'+ticker+'_'+str(hidden_dim)+'_' + train_from + '_' + train_until + '_'+str(num_epochs) + '_no_embeddings_warm_start_attention_retrained_while_testing')

predictions_df['Predictions'].value_counts()

predictions_df['Actual_Value'].value_counts()


# from sklearn.metrics import classification_report

# print(classification_report(predictions_df['Actual_Value'], predictions_df['Predictions']))
