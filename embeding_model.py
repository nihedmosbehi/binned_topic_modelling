import gensim.downloader as api
from transformers import BertTokenizer, BertModel
import torch

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

word2vec_model = api.load("word2vec-google-news-300")

def getModel():
    # return word2vec_model
    return model