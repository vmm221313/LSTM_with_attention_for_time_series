import pandas as po
from ray import tune
import ray
from ray.tune import track
track.init()
import os
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from customLSTM import baseLSTM
from trainer import train
from tester import test


def perform_gridSearch(ticker, window_size, train_from, train_until, model, loss_function, optimizer, num_epochs, input_dim, num_output_classes, hidden_dim, retrain_while_testing, retrain_after,):
    def gridSearch(config):
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = config['lr']
            
        num_epochs = config['num_epochs']
        
        for i in range(3):
            model, (hidden_state, cell_state) = train(ticker, window_size, train_from, train_until, model, loss_function, optimizer, num_epochs, input_dim, num_output_classes, hidden_dim)

            predictions_df = test(ticker, window_size, test_from, test_until, retrain_while_testing, retrain_after, model, hidden_state, cell_state, loss_function, optimizer, num_epochs, input_dim, num_output_classes, hidden_dim)
            f1 = f1_score(predictions_df['Actual_Value'], predictions_df['Predictions'],average='weighted')

            tune.track.log(mean_accuracy = f1)

    analysis = tune.run(gridSearch, config = {'lr': tune.grid_search([0.001, 0.01, 0.05, 0.1, 0.5]),
                                              'num_epochs': tune.grid_search([5, 10, 15, 20]),
                                              'dropout': tune.grid_search([0.1, 0.2, 0.3])
                                             })
    print("Best config: ", analysis.get_best_config(metric = "mean_accuracy"))

    return analysis
