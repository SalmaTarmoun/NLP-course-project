import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.utils.rnn 


class Encoder(nn.Module):

    def __init__(self, 
            emb_size, 
            hidden_size, 
            p_dropout): 
    
            super(Encoder, self).__init__()
            self.emb_size = emb_size
            self.hidden_size = hidden_size
            
            self.p_dropout = p_dropout
       
            self.lstm = nn.LSTM(self.emb_size, self.hidden_size)
            self.dropout_layer = nn.Dropout(self.p_dropout) 

            self.init_weights()
             
    def init_weights(self):
        init.uniform(self.lstm.weight_ih_l0, a = -0.01, b = 0.01)
        init.orthogonal(self.lstm.weight_hh_l0)
        self.lstm.weight_ih_l0.requires_grad = True
        self.lstm.weight_hh_l0.requires_grad = True      
            
    def forward(self, embeddings):
        _, (last_hidden, _) = self.lstm(embeddings) 
        last_hidden = self.dropout_layer(last_hidden[-1])

        return last_hidden

    
class DualEncoder(nn.Module):
     
    def __init__(self, encoder):
        super(DualEncoder, self).__init__()
        self.encoder = encoder
        self.hidden_size = self.encoder.hidden_size
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)     
        init.xavier_normal(M)
        self.M = nn.Parameter(M, requires_grad = True)

    def forward(self, context_tensor, response_tensor):
        
        context_last_hidden = self.encoder(context_tensor) 
        response_last_hidden = self.encoder(response_tensor) 
        
        context = context_last_hidden.mm(self.M)
        
        context = context.view(-1, 1, self.hidden_size) 
        
        response = response_last_hidden.view(-1, self.hidden_size, 1)
        
        score = torch.bmm(context, response).view(-1, 1)
    
        return score
