import torch
from utils import shuffle_dataframe
import torch.autograd as autograd
import random 
from utils import increase_count, get_embeddings
import numpy as np




def train(epoch, training_dataframe, embed_dict, dual_encoder, optimizer, loss_func, device):
    shuffle_dataframe(training_dataframe)         
    sum_loss_training = 0.0
    training_correct_count = 0
    dual_encoder.train()
    
    for index, row in training_dataframe.iterrows():
        
        if row["Label"]==1:
          row={"Context":row["Context"],"Utterance":row["Correct"],"Label":1}
        else:
          row={"Context":row["Context"],"Utterance":random.choice(list(row)[3:]),"Label":0}
        context = get_embeddings(embed_dict, row['Context'])
        response = get_embeddings(embed_dict, row['Utterance'])
        label = row['Label']
        label = np.array(label).astype(np.float32)
        
        context = autograd.Variable(torch.FloatTensor(context).view(len(context),1, -1), requires_grad = False).to(device)
        response = autograd.Variable(torch.FloatTensor(response).view(len(response), 1, -1), requires_grad = False).to(device)
        label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label).reshape(1,1))), requires_grad = False).to(device)
        score = dual_encoder(context, response)
        loss = loss_func(score, label)
        sum_loss_training += loss.data
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        training_correct_count = increase_count(training_correct_count, torch.sigmoid(score), label)
                                            
    training_accuracy = training_correct_count/len(training_dataframe)
    train_loss = sum_loss_training/len(training_dataframe)  
    return training_accuracy, train_loss      



def val(epoch, validation_dataframe, embed_dict, dual_encoder, optimizer, loss_func, device):
     
    shuffle_dataframe(validation_dataframe)
    
    validation_correct_count = 0

    sum_loss_validation = 0.0

    dual_encoder.eval()

    for index, row in validation_dataframe.iterrows():
        if row["Label"]==1:
          row={"Context":row["Context"],"Utterance":row["Correct"],"Label":1}
        else:
          row={"Context":row["Context"],"Utterance":random.choice(list(row)[3:]),"Label":0}
        
        context = get_embeddings(embed_dict, row['Context'])
        response = get_embeddings(embed_dict, row['Utterance'])
        label = float(row['Label'])
        
        context = autograd.Variable(torch.FloatTensor(context).view(len(context),1, -1), requires_grad = False).to(device)
        response = autograd.Variable(torch.FloatTensor(response).view(len(response), 1, -1), requires_grad = False).to(device)
                        
        label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label).reshape(1,1)))).to(device)
        score = dual_encoder(context, response)
        loss = loss_func(score, label)
        
        sum_loss_validation += loss.data
        
        validation_correct_count = increase_count(validation_correct_count, torch.sigmoid(score), label)
            
    validation_accuracy = validation_correct_count/len(validation_dataframe)
    val_loss = sum_loss_validation/len(validation_dataframe)
    
    return validation_accuracy, val_loss
                
    
    