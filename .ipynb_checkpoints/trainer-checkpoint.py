import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from dataloader import get_data_for_given_ticker
from tqdm import tqdm
import pandas as po

def train(ticker, window_size, train_from, train_until, model, loss_function, optimizer, num_epochs, input_dim, num_output_classes, hidden_dim):
    
    train_df, targets, dates = get_data_for_given_ticker(ticker, input_dim, start_date = train_from, end_date = train_until, train = True)
    
    ticker_embeddings_df = po.read_csv('more_data/ticker_embeddings.csv')
    embedding = ticker_embeddings_df[ticker_embeddings_df['#RIC'] == ticker]
    del ticker_embeddings_df
    embedding = embedding.drop('#RIC', axis = 1)
    embedding *= 10000
   
    hidden_state = torch.randn(1, 1, hidden_dim)
    embedding_tensor = torch.tensor(embedding.to_numpy()).reshape(1, 1, len(embedding.columns))
    hidden_state[:, :, :len(embedding.columns)] = embedding_tensor
    
    cell_state = torch.randn(1, 1, hidden_dim)
    
    for epoch in range(num_epochs):
        for i in tqdm(range(window_size, len(train_df))):
            model.zero_grad()
            
            #print(train_df[i-window_size:i][0]) #print this to check if the rolling windows are working correctly
            
            input_ = torch.tensor(train_df[i-window_size:i].to_numpy(), dtype = torch.float).view(window_size, 1, input_dim)
            prediction, (hidden_state, cell_state) = model(input_, hidden_state, cell_state)
            
            target = torch.tensor([targets[i]], dtype = torch.long)
            prediction = prediction + 10**(-8)
            hidden_state.detach_()
            cell_state.detach_()

            loss = loss_function(prediction, target)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
         
        print(loss)
           
    return model, (hidden_state, cell_state)



