import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')

def text_embedding(sentences):
    sentence_embeddings = model.encode(sentences)
    return sentence_embeddings

