import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import DNN
import os

os.chdir("/Users/wzq/Desktop/game/ITI5212_Rec_Sys_2024")
print(os.getcwd())  # 查看当前工作目录


train_data = pd.read_csv('./data/train_processed.csv')
train_data['rating'] = train_data['rating'] - 1

# 定义特征和目标变量
features = ['user_rating_mean', 'user_rating_count', 'user_votes_mean', 'user_votes_max', 'user_helpful_votes_mean', 'user_helpful_votes_max'] \
         + ['product_id', 'product_rating_mean', 'product_rating_count', 'product_votes_mean', 'product_votes_max','product_helpful_votes_mean', 'product_helpful_votes_max']
# features = ['user_rating_mean', 'user_votes_mean', 'product_rating_mean', 'product_votes_mean']

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
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)

# 实例化模型
input_size = len(features)
hidden_size = 64
output_size = 5
model = DNN(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100

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
    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        print(outputs)
        _, predicted = torch.max(outputs, 1)
        print(labels.size())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acc = 100 * correct / total
print(f"Test Accuracy: {acc:.2f}%")






