# -*- coding: utf-8 -*-

import numpy as np
from preprocessing import preprocess


def create_embed_dict(glovefile):
    """
    Reads glove embedding file and creates a dictionary with (key, value) = (word, embedding)
    """
    lines = open(glovefile, 'r', encoding="utf8").readlines()
    embed_dic = {}
    
    for line in lines:
        word = line.split()[0]
        vector = np.array(line.split()[1:], dtype='float32') #32
        embed_dic[word] = vector
    return embed_dic

def increase_count(correct_count, score, label):
    """
    Increases the count of correct answers by 1 if the score (probability) is higher 
    than 0.5 and the label is 1 or the probability is lower than 0.5 and the label is 0
    """
    if ((score.data[0][0] >= 0.5) and (label.data[0][0] == 1.0)) or ((score.data[0][0] < 0.5) and (label.data[0][0]  == 0.0)):
       correct_count +=1  
   
    return correct_count

def shuffle_dataframe(dataframe):
    dataframe.reindex(np.random.permutation(dataframe.index))
    
def get_embeddings(embed_dict, sentence):
    """Returns the Glove embeddings of a sentence
    """
    
    sentence = preprocess(sentence).split()
    embeds = []
    emb_dim = len(list(embed_dict.values())[0])
    for word in sentence:
        if word == 'the':
           pass
        else:
            try:
                embeds.append(embed_dict[word])
            except Exception:
                embeds.append(np.zeros(emb_dim,))
                
    embed = np.vstack(embeds)
    return embed