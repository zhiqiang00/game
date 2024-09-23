import os
import time
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from model import DNN
from utils import save_zip_file

os.chdir("/Users/wzq/Desktop/game/ITI5212_Rec_Sys_2024")
print(os.getcwd())  # 查看当前工作目录


train_data = pd.read_csv('./data/train_processed.csv')
train_data['rating'] = train_data['rating'] - 1

# 定义特征和目标变量
# features = ['user_rating_mean', 'user_rating_count', 'user_votes_mean', 'user_votes_max', 'user_helpful_votes_mean', 'user_helpful_votes_max'] \
#          + ['product_id', 'product_rating_mean', 'product_rating_count', 'product_votes_mean', 'product_votes_max','product_helpful_votes_mean', 'product_helpful_votes_max']
# features = ['user_rating_mean', 'user_votes_mean', 'product_rating_mean', 'product_votes_mean']
features = ['user_rating_mean', 'user_rating_count', 'user_votes_mean', 'user_votes_max', 'user_helpful_votes_mean',] \
         + ['product_id', 'product_rating_mean', 'product_rating_count', 'product_votes_mean','product_helpful_votes_mean']

X = train_data[features]
y = train_data['rating']


# 归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 将数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.long)

# 使用 DataLoader 加载数据
train_dataset = TensorDataset(X_train,y_train)
test_dataset = TensorDataset(X_test,y_test)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# 实例化模型
input_size = len(features)
hidden_size = 64
output_size = 5
model = DNN(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 早停的参数
patience = 20  # 设置耐心，表示如果经过 `patience` 个 epoch 没有提升则停止训练
best_loss = float('inf')  # 用于存储最佳损失
early_stop_counter = 0  # 用于计数没有改进的 epoch 次数
best_model_state = None  # 用于保存最优模型的状态

n_epochs = 1000
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {running_loss/len(train_loader):.4f}")


    model.eval()
    val_loss = 0.0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(test_loader)
    print(f"Epoch [{epoch+1}/{n_epochs}], Validation Loss: {avg_val_loss:.4f}")
    rmse = np.sqrt(mean_squared_error(true_labels, predictions))
    print(f"Epoch [{epoch+1}/{n_epochs}], RMSE Loss: {rmse:.4f}")

    # 早停判断
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        early_stop_counter = 0
        best_model_state = model.state_dict() # 保存最优模型状态
    else:
        early_stop_counter += 1
        print(f"Early stopping counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# 加载最优模型
if best_model_state:
    model.load_state_dict(best_model_state)

# 在测试数据上预测并计算 RMSE
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    print(f"=========infer========={time.strftime('%Y%m%d%H%M', time.localtime())}")
    # 预测评分
    test_data = pd.read_csv('./data/test_processed.csv')
    # print(test_data.shape)
    inputs = test_data[features]
    # 归一化
    scaler = StandardScaler()
    inputs = scaler.fit_transform(inputs)
    inputs = torch.tensor(inputs, dtype=torch.float32)

    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    # print(predicted.shape)
    test_data['rating'] = predicted.cpu().numpy()
    submission = test_data[['ID', 'rating']]
    save_zip_file(submission, )
    print(f"=========infer done========={time.strftime('%Y%m%d%H%M', time.localtime())}")






