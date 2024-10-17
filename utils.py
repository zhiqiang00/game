import time
import numpy as np
import pandas as pd
import os
import zipfile
from nltk.tokenize import word_tokenize

def calculate_mrr(df_sorted, df_clicked):
    mrr_score = 0.0
    total_queries = len(df_clicked)
    for index, row in df_clicked.iterrows():
        # 获取当前用户的所有推荐
        recommendations = df_sorted[df_sorted['uid'] == row['uid']].reset_index(drop=True).head(5)
        # 检查点击的视频是否在推荐列表中，并且计算其排名
        if row['vid'] in recommendations['vid'].values:
            rank = recommendations[recommendations['vid'] == row['vid']]['vid'].index[0] + 1
            mrr_score += 1 / rank
    # 计算平均MRR
    return mrr_score / total_queries if total_queries > 0 else 0
# 计算MRR
# mrr = calculate_mrr(df_recommendations, df_clicked)
# print(f"MRR: {mrr}")

def save_zip_file(submission):
    file_name = f"./data/submit_example_A{time.strftime('%Y%m%d%H%M', time.localtime())}.csv"
    submission.to_csv(f"./data/submit_example_A{time.strftime('%Y%m%d%H%M', time.localtime())}.csv",index=False, encoding='utf-8')
    file_name_zip = f"./data/submit_example_A{time.strftime('%Y%m%d%H%M', time.localtime())}.csv.zip"

    with zipfile.ZipFile(file_name_zip, 'w') as z:
        z.write(file_name, os.path.basename(file_name))

def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding_vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding_vector
    return embeddings_index

# 分词并加载词嵌入
def sentence_to_embedding(sentence, embeddings_index, enable_mean=False, embedding_dim=50):
    words = word_tokenize(sentence.lower())  # 分词，并转化为小写
    embeddings = []
    for word in words:
        if word in embeddings_index:
            embeddings.append(embeddings_index[word])
        else:
            embeddings.append(np.zeros(embedding_dim))  # 若词不在 GloVe 中，则用零向量替代
    if enable_mean:
        np.mean(embeddings, axis=0)
    return np.array(embeddings)