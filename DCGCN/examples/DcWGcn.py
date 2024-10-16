
from rec4torch.models import DeepCrossingWithGCN
from rec4torch.models import GraphConvolution  # 导入你的图卷积网络模块，假设为 GCN
# import pandas as pd
# from torch import nn, optim
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import LabelEncoder
# from rec4torch.inputs import SparseFeat, VarLenSparseFeat, build_input_array
# from rec4torch.models import DeepCrossing
# from rec4torch.snippets import sequence_padding, seed_everything
# import random
#
import numpy as np
import pandas as pd
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from rec4torch.inputs import SparseFeat, VarLenSparseFeat, build_input_array
from rec4torch.models import DeepCrossing
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

    data = pd.read_csv("./datasets/movies.csv",encoding='gbk')
    sparse_features = ["movie_id", "user_id", "gender", "timestamp", "title",]

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

    # 生成训练样本
    train_X, train_y = build_input_array(data, linear_feature_columns + dnn_feature_columns, target=['rating'])
    train_X = torch.tensor(train_X, device=device)
    train_y = torch.tensor(train_y, dtype=torch.float, device=device)
    # 构建邻接矩阵
    user_ids = data['user_id'].unique()
    movie_ids = data['movie_id'].unique()
    # 构建空白的邻接矩阵
    adjacency_matrix = np.zeros((len(user_ids), len(movie_ids)))
    # 填充邻接矩阵，根据用户对电影的评分情况
    for _, row in data.iterrows():
        user_idx = np.where(user_ids == row['user_id'])[0][0]
        movie_idx = np.where(movie_ids == row['movie_id'])[0][0]
        adjacency_matrix[user_idx, movie_idx] = row['rating']

    # 转换为 PyTorch 张量
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float, device=device)

    print('这是 adj', adjacency_matrix)
    # adjacency_matrix = np.random.rand(train_X.shape[0], train_y.shape[0])  #邻接矩阵
    # adjacency_matrix = torch.tensor(adjacency_matrix, device=device)
    # print('zhegeshi adj',adjacency_matrix)
    return train_X, train_y, linear_feature_columns, dnn_feature_columns, adjacency_matrix


# 加载数据集
train_X, train_y, linear_feature_columns, dnn_feature_columns, adjacency_matrix= get_data()
train_dataloader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
print('zhege shi ',linear_feature_columns)
print('这是 adj', adjacency_matrix)
# 模型定义
deep_crossing_model = DeepCrossing(linear_feature_columns, dnn_feature_columns)
gcn_model = GraphConvolution(input_dim=10, output_dim=2, linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns)# 根据你的 GCN 模型定义输入和输出维度
model = DeepCrossingWithGCN(linear_feature_columns, dnn_feature_columns, deep_crossing_model, gcn_model, adjacency_matrix, gcn_input_dim=10, gcn_hidden_dim=64, gcn_output_dim=2)  # DeepCrossingWithGCN 是将 Deep Crossing 和 GCN 结合起来的模型

model.to(device)

model.compile(
    loss=nn.MSELoss(),
    optimizer=optim.Adam(model.parameters(), lr=1e-2),
    metrics=['mse', 'acc', 'mae'],
)

if __name__ == "__main__":
    model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[])
