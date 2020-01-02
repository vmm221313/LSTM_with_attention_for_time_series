import torch.optim as optim
from ray import tune
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from customLSTM import baseLSTM
from trainer import train
from tester import test
import os
import pandas as po
from sklearn.metrics import f1_score


input_dim = 23
num_output_classes = 5
hidden_dim = 2000
num_epochs = 1
window_size = 5
batch_size = 1 #number of tickers to be passed at the same time

ticker = 'CLc1'
train_from = '2009-01-01' # >= applied
train_until = '2009-02-01' # <= applied
test_from = '2009-02-02' # >= applied
test_until = '2009-02-10' # <= applied

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

import os
os.getcwd()

os.listdir('merged_data_without_embeddings/')

 df = po.read_csv('/Users/VarunMadhavan/Desktop/Notes/NLP/ISB/LSTM/LSTM_attention/merged_data_without_embeddings/' + ticker + 'merged.csv')

os.chdir('/Users/VarunMadhavan/Desktop')


def gridd(config):
    input_dim = 23
    num_output_classes = 5
    hidden_dim = 2000
    num_epochs = 1
    window_size = 5
    batch_size = 1
    
    ticker = 'CLc1'
    
    os.chdir('/Users/VarunMadhavan/Desktop/Notes/NLP/ISB/LSTM/LSTM_attention')
    df = po.read_csv('merged_data_without_embeddings/' + ticker + 'merged.csv')
    
    df = df[df['No. Trades'] != 0].reset_index(drop = True)
    val_cts = df['class'].value_counts().to_numpy()
    del df 
    
    val_cts = val_cts/sum(val_cts)
    weights = torch.tensor(val_cts, dtype = torch.float)
    
#weights
    train_from = '2009-01-01' # >= applied
    train_until = '2009-02-01' # <= applied
    test_from = '2009-02-02' # >= applied
    test_until = '2009-02-10' # <= applied
    
    model = baseLSTM(input_dim, hidden_dim, num_output_classes, window_size)
    loss_function = nn.CrossEntropyLoss(ignore_index = 5, weight = weights)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    for i in range(3):
         
        model, (hidden_state, cell_state) = train(ticker, window_size, train_from, train_until, model, loss_function, optimizer, num_epochs, input_dim, num_output_classes, hidden_dim)
        predictions_df = test(ticker, window_size, test_from, test_until, input_dim, model, hidden_state, cell_state)
    
        acc=f1_score(predictions_df['Actual_Value'], predictions_df['Predictions'],average='weighted')
        #acc=test()
        tune.track.log(mean_accuracy=acc)


analysis = tune.run(gridd,config={"lr": tune.grid_search([0.1,0.3])})

print("Best config: ",analysis.get_best_config(metric="mean_accuracy"))

analysis = tune.run(train,config={"num_epochs": tune.grid_search([5,10,15,20,25])})

print("Best config: ",analysis.get_best_config(metric="mean_accuracy"))

analysis = tune.run(train,config={"dropout": tune.grid_search([0.1,0.2,0.3])})

print("Best config: ",analysis.get_best_config(metric="mean_accuracy"))

analysis.dataframe()

# !ray 

import ray
ray.tune.track.init()


ray.tune.track.init()

track.log

from ray import tune
import ray
from ray.tune import track
track.init()


