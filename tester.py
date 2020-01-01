import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from dataloader import get_data_for_given_ticker
from tqdm import tqdm
import pandas as po
from sklearn.metrics import classification_report


def test(ticker, window_size, test_from, test_until, input_dim, model, hidden_state, cell_state):
    
    test_df, targets = get_data_for_given_ticker(ticker, input_dim, start_date = test_from, end_date = test_until, train = False)

    predictions_df = po.DataFrame()
    predictions = []
    actual_values = []
    
    for i in tqdm(range(window_size, len(test_df))):
        with torch.no_grad():
            input_ = torch.tensor(test_df[i-window_size:i].to_numpy(), dtype = torch.float).view(window_size, 1, input_dim)
            prediction, (hidden_state, cell_state) = model(input_, hidden_state, cell_state)

            prediction = nn.Softmax()(prediction)[0].numpy().argmax()
            predictions.append(prediction)
            actual_values.append(targets[i])

    predictions_df['Predictions'] = predictions
    predictions_df['Actual_Value'] = actual_values
    
    print(classification_report(predictions_df['Actual_Value'], predictions_df['Predictions']))
    
    return predictions_df
