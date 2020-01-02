import pandas as po
from hyperopt import fmin, tpe, hp, STATUS_OK 

# from ray import tune
# import ray
# from ray.tune import track
# track.init()

import os
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from customLSTM import baseLSTM
from trainer import train
from tester import test


def perform_gridSearch(ticker, window_size, train_from, train_until, model, loss_function, optimizer, num_epochs, input_dim, num_output_classes, hidden_dim, retrain_while_testing, retrain_after, dropout_prob):
    best_f1 = 0
    
    def gridSearch(space):
        global best_lr, best_num_epochs, best_dropout_prob, best_f1
        model_m=model
        lr = space['lr']
        num_epochs = space['num_epochs']
        dropout_prob = space['dropout_prob']

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(window_size)
        model_m, (hidden_state, cell_state) = train(ticker, window_size, train_from, train_until, model_m, loss_function, optimizer, num_epochs, input_dim, num_output_classes, hidden_dim, dropout_prob)
        predictions_df = test(ticker, window_size, test_from, test_until, retrain_while_testing, retrain_after, model, hidden_state, cell_state, loss_function, optimizer, num_epochs, input_dim, num_output_classes, hidden_dim)
        f1 = f1_score(predictions_df['Actual_Value'], predictions_df['Predictions'],average='weighted')

        if (f1 > best_f1):
            best_lr = lr
            best_num_epochs = num_epochs
            best_dropout_prob = dropout_prob
            best_f1 = f1
            print('best_lr = {}'.format(best_lr))
            print('best_num_epochs = {}'.format(num_epochs))
            print('best_dropout_prob {}'.format(dropout_prob))
            print('best_f1 = {}'.format(f1))

        return {'loss': f1, 'status': STATUS_OK }
    
    space = {
    'lr': hp.uniform('lr', 0.001, 0.1),
    'num_epochs': hp.uniform('num_epochs', 5, 10),
    'dropout_prob': hp.uniform('dropout_prob', 0.1, 0.2)
    }

    best_scores = fmin(fn=gridSearch, space=space, algo=tpe.suggest, max_evals=3)
    
    return best_scores

# def perform_gridSearch(ticker, window_size, train_from, train_until, model, loss_function, optimizer, num_epochs, input_dim, num_output_classes, hidden_dim, retrain_while_testing, retrain_after, dropout_prob):
#     
#     def gridSearch(config):
#         model_m=model
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = config['lr']
#             
#         num_epochs = config['num_epochs']
#         dropout_prob = config['dropout']
#         
#         
#         for i in range(3):
#             model_m, (hidden_state, cell_state) = train(ticker, window_size, train_from, train_until, model_m, loss_function, optimizer, num_epochs, input_dim, num_output_classes, hidden_dim, dropout_prob)
#
#             predictions_df = test(ticker, window_size, test_from, test_until, retrain_while_testing, retrain_after, model, hidden_state, cell_state, loss_function, optimizer, num_epochs, input_dim, num_output_classes, hidden_dim)
#             f1 = f1_score(predictions_df['Actual_Value'], predictions_df['Predictions'],average='weighted')
#
#             tune.track.log(mean_accuracy = f1)
#
#     '''
#     analysis = tune.run(gridSearch, config = {'lr': tune.grid_search([0.001, 0.01, 0.05, 0.1, 0.5]),
#                                               'num_epochs': tune.grid_search([5, 10, 15, 20]),
#                                               'dropout': tune.grid_search([0.1, 0.2, 0.3])
#                                              })
#                                              
#     '''
#     analysis = tune.run(gridSearch, config = {'lr': tune.grid_search([0.001, 0.1]),
#                                           'num_epochs': tune.grid_search([5, 10]),
#                                           'dropout': tune.grid_search([0.1, 0.2])
#                                          })
#     
#     
#     
#     print("Best config: ", analysis.get_best_config(metric = "mean_accuracy"))
#
#     return analysis


