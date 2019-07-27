import argparse
import numpy as np
import torch.autograd as autograd
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import create_embed_dict, get_embeddings
from model import Encoder, DualEncoder
import torch
from train_val import train, val


class DialogueManager:
    def __init__(self, glovefile="glove.6B.200d.txt"):
        self.embed_dict = create_embed_dict(glovefile)
        

    def load(self,path, emb_dim, hidden_size, p_dropout):
        encoder, dual_encoder = self.create_model(emb_dim, hidden_size, p_dropout)
        device = torch.device('cuda:0' if torch.cuda.is_avaliable() else 'cpu')
        dual_encoder.load_state_dict(torch.load(path, map_location=device))
        dual_encoder.eval()
        self.dual_encoder = dual_encoder

    def save(self,path):
        torch.save(self.dual_encoder.state_dict(), path)

    def initialize_df(self, data):
        """
        Initializes the dataframe that we will use for training:
        the columns of this dataframe are :
            Context : corresponds to the utterance
            Correct : The correct answer
            Wrong1, Wrong2,.., Wrong19: the 19 wrong responses
            Label : 0 or 1
        """
        Questions = []
        CorrectAnswers=[]
        WrongAnswers=[]
        # We extract the correct and wrong answers
        for _,_, dialogue in data:
            for idx,question,correct,options in dialogue:
                wa=[]
                Questions.append(question)
                i=0
                for answer in options:
                  if answer==correct:
                    if i==0:
                      CorrectAnswers.append(answer)
                      i+=1
                  else:
                    wa.append(answer)
                if len(wa)<19:
                  for i in range(0,19-len(wa)):
                    wa.append(wa[0])
                WrongAnswers.append(wa)
                
        df_wa = pd.DataFrame(WrongAnswers, columns = ['Wrong{}'.format(i+1) for i in range(19)])
        # We duplicate the rows so that each question can be processed twice in the training,
        # once with the correct answer (label 1) and once with one of the wrong answers(label 0)
        label0=[0]*len(Questions)
        df0 = pd.DataFrame({"Context":Questions, "Correct":CorrectAnswers,"Label":label0})
        label1=[1]*len(Questions)
        df1 = pd.DataFrame({"Context":Questions, "Correct":CorrectAnswers,"Label":label1})
        df0=pd.concat([df0, df_wa], axis=1)
        df1=pd.concat([df1, df_wa], axis=1)
        df=pd.concat([df1,df0],axis=0,ignore_index=True) 
        df.reindex(np.random.permutation(df.index)) # shuffle the dataframe
        self.df = df
        
    def create_model(self, emb_dim, hidden_size, p_dropout):
        """
        Initializes the dual encoder
        :param emb_dim : embedding dimension
        :param hidden size : the size of the hidden states
        :param p_drouput : the dropout probability
        
        """
        
        encoder = Encoder(
            emb_size = emb_dim,
            hidden_size = hidden_size,
            p_dropout = p_dropout)

        dual_encoder = DualEncoder(encoder)
        return encoder, dual_encoder
        
        
    def train(self,data, emb_dim, hidden_size, p_dropout, lr, l2_penalty, epochs):
        
        device = torch.device('cuda:0' if torch.cuda.is_avaliable() else 'cpu')
        encoder, dual_encoder = self.create_model(emb_dim, hidden_size, p_dropout)
        encoder.to(device)
        dual_encoder.to(device)
        self.initialize_df(data)  
        training_dataframe, validation_dataframe = train_test_split(self.df, test_size=0.05)
        
        optimizer = torch.optim.Adam(dual_encoder.parameters(), lr = lr, weight_decay = l2_penalty)
        
           
        loss_func = torch.nn.BCEWithLogitsLoss()
        loss_func.to(device)
        
        for epoch in range(epochs):
            train(epoch, training_dataframe, self.embed_dict, dual_encoder, optimizer, loss_func, device)
            val(epoch, validation_dataframe, self.embed_dict, dual_encoder, optimizer, loss_func, device)
        self.dual_encoder = dual_encoder

    def findBest(self,utterance,options):
        """
        finds the best utterance out of all those given in options
        :param utterance: a single string
        :param options: a sequence of strings
        :return: returns one of the strings of options
        """
        self.dual_encoder.eval()
        device = torch.device('cuda:0' if torch.cuda.is_avaliable() else 'cpu')
        scores = []
        context_embed = get_embeddings(self.embed_dict, utterance)
        context = autograd.Variable(torch.FloatTensor(context_embed).view(len(context_embed),1, -1), requires_grad = False).to(device)
                
        for answer in options:
            response = get_embeddings(self.embed_dict, answer)
            response = autograd.Variable(torch.FloatTensor(response).view(len(response), 1, -1), requires_grad = False).to(device)
            score = self.dual_encoder(context, response)
            scores.append(score)
        pred = np.argmax(scores) # pick the answer with the highest score
        return options[pred]


def loadData(path):
    """
        :param path: containing dialogue data of ConvAI (eg:  train_both_original.txt, valid_both_original.txt)
        :return: for each dialogue, yields (description_of_you, description_of_partner, dialogue) where a dialogue
            is a sequence of (utterance, answer, options)
    """
    with open(path) as f:
        descYou, descPartner = [], []
        dialogue = []
        for l in f:
            l=l.strip()
            lxx = l.split()
            idx = int(lxx[0])
            if idx == 1:
                if len(dialogue) != 0:
                    yield descYou,  descPartner, dialogue
                # reinit data structures
                descYou, descPartner = [], []
                dialogue = []

            if lxx[2] == 'persona:':
                # description of people involved
                if lxx[1] == 'your':
                    description = descYou
                elif lxx[1] == "partner's":
                    description = descPartner
                else:
                    assert 'Error, cannot recognize that persona ({}): {}'.format(lxx[1],l)
                description.append(lxx[3:])

            else:
                # the dialogue
                lxx = l.split('\t')
                utterance = ' '.join(lxx[0].split()[1:])
                answer = lxx[1]
                options = [o for o in lxx[-1].split('|')]
                dialogue.append( (idx, utterance, answer, options))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to model file (for saving/loading)', required=True)
    parser.add_argument('--text', help='path to text file (for training/testing)', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--test', action='store_true')
    parser.add_argument('--gen', help='enters generative mode')
    parser.add_argument('--glove_path', help='path to glove embeddings', default="glove.6B.200d.txt")

    opts = parser.parse_args()

    dm = DialogueManager(opts.glove_path)
    if opts.train:
        text = loadData(opts.text)
        dm.train(data=text, emb_dim=200, hidden_size=200, p_dropout=0.25,
                 lr=0.0001, l2_penalty=0.0001, epochs=50)
        dm.save(opts.model)
    else:
        assert opts.test,opts.test
        dm.load(opts.model, emb_dim=200, hidden_size=200, p_dropout=0.25)
        for _,_, dialogue in loadData(opts.text):
            for idx, utterance, answer, options in dialogue:
                print(idx,dm.findBest(utterance,options))

