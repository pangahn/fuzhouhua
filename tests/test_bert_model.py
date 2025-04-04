# -*- coding: utf-8 -*-
import torch
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer


def bert_similarity(text1, text2):
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertModel.from_pretrained("bert-base-chinese")

    inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    sentence_vec1 = outputs1.last_hidden_state[:, 0, :]
    sentence_vec2 = outputs2.last_hidden_state[:, 0, :]

    sim = torch.cosine_similarity(sentence_vec1, sentence_vec2).item()
    return sim


def sentence_transformer_similarity(model_name, text1, text2):
    """
    text1 = "说慢着我去取钱"
    text2 = "我怎么办"


    Original models
    ----------------
        | model_name                           | score |
        |--------------------------------------|-------|
        | all-MiniLM-L12-v2                    | 0.952 |
        | all-MiniLM-L6-v2                     | 0.952 |
        | all-mpnet-base-v2                    | 0.938 |
        | all-distilroberta-v1                 | 0.567 |
        | distiluse-base-multilingual-cased-v2 | 0.275 |

    Community models
    ----------------
        | model_name                           | score |
        |--------------------------------------|-------|
        | infgrad/stella-base-zh-v2            | 0.676 |
        | BAAI/bge-m3                          |  |
    """
    model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    embeddings = model.encode([text1, text2], normalize_embeddings=True)
    similarity = model.similarity(embeddings[0], embeddings[1])
    return float(similarity[0][0])


text1 = "说慢着我去取钱"
text2 = "我怎么办"

model_name = "BAAI/bge-m3"
sim = sentence_transformer_similarity(model_name, text1, text2)
print(sim)
