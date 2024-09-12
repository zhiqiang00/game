import numpy as np
import pandas as pd

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