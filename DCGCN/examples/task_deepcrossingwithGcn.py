import random

import numpy as np
import pandas as pd
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from rec4torch.inputs import SparseFeat, VarLenSparseFeat, build_input_array
from rec4torch.models import DeepCrossingWithGCN
from rec4torch.snippets import sequence_padding, seed_everything

batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(42)


def get_data():
    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    data = pd.read_csv("./datasets/movielens_sample.txt")
    sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip", ]
    user_movie_interaction = pd.pivot_table(data, values='rating', index='user_id', columns='movie_id', fill_value=0)

    # 构造邻接矩阵
    adj_matrix = user_movie_interaction.values
    # 离散变量编码
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 序列特征处理
    key2index = {}
    genres_list = list(map(split, data['genres'].values))
    genres_list = sequence_padding(genres_list)
    data['genres'] = genres_list.tolist()

    # 离散特征和序列特征处理
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4) for feat in sparse_features]

    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
        key2index) + 1, embedding_dim=4), maxlen=genres_list.shape[-1], pooling='mean')]

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    # 创建用于GCN的特征列
    gcn_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4) for feat in ['user_id', 'movie_id']]

    # 生成训练样本
    train_X, train_y = build_input_array(data, linear_feature_columns + dnn_feature_columns, target=['rating'])
    train_X = torch.tensor(train_X, device=device)
    train_y = torch.tensor(train_y, dtype=torch.float, device=device)
    return train_X, train_y, linear_feature_columns, dnn_feature_columns, gcn_feature_columns,adj_matrix

# 加载数据集
train_X, train_y, linear_feature_columns, dnn_feature_columns, gcn_feature_columns,adj_matrix= get_data()
train_dataloader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)

# 模型定义
model = DeepCrossingWithGCN(linear_feature_columns, dnn_feature_columns, gcn_feature_columns=gcn_feature_columns)
model.to(device)
def eval(y_pred, y_true):
# 仅做示意
    return {'rouge-1': random.random(),'rouge-2': random.random(),'bleu': random.random()}

model.compile(
    loss=nn.MSELoss(),
    optimizer=optim.Adam(model.parameters(), lr=1e-2),
    metrics=[eval, 'mse','acc'],
)

if __name__ == "__main__":
    model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[])
