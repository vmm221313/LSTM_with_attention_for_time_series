import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from dataloader import get_data_for_given_ticker
from trainer import train
from tqdm import tqdm
import pandas as po
from sklearn.metrics import classification_report


def test(ticker, window_size, test_from, test_until, retrain_while_testing, retrain_after, model, hidden_state, cell_state, loss_function, optimizer, num_epochs, input_dim, num_output_classes, hidden_dim):
    
    test_df, targets, dates = get_data_for_given_ticker(ticker, input_dim, start_date = test_from, end_date = test_until, train = False)

    predictions_df = po.DataFrame()
    predictions = []
    actual_values = []
    
    for i in tqdm(range(window_size, len(test_df))):
        with torch.no_grad():
            input_ = torch.tensor(test_df[i-window_size:i].to_numpy(), dtype = torch.float).view(window_size, 1, input_dim)
            prediction, (hidden_state, cell_state) = model(input_, hidden_state, cell_state)

            prediction = nn.Softmax(dim=1)(prediction)[0].numpy().argmax()
            predictions.append(prediction)
            actual_values.append(targets[i])
    
        if retrain_while_testing and i%retrain_after == 0:
            train_from = dates[i-window_size+1]
            train_until = dates[i]
            model, (hidden_state, cell_state) = train(ticker, window_size, train_from, train_until, model, loss_function, optimizer, num_epochs, input_dim, num_output_classes, hidden_dim)

            
    predictions_df['Predictions'] = predictions
    predictions_df['Actual_Value'] = actual_values
    
    print(classification_report(predictions_df['Actual_Value'], predictions_df['Predictions']))
    
    return predictions_df, model
