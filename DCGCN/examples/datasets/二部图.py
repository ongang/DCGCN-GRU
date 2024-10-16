import matplotlib.pyplot as plt
import networkx as nx

# 示例数据
user_set = {'user1', 'user2', 'user3'}
movie_set = {'movie1', 'movie2', 'movie3'}
ratings = {('user1', 'movie1'): 5, ('user2', 'movie2'): 4, ('user3', 'movie3'): 3}

# 创建图对象
G = nx.Graph()

# 添加用户节点
G.add_nodes_from(user_set, bipartite=0)

# 添加电影节点
G.add_nodes_from(movie_set, bipartite=1)

# 添加边以及权重
for (user, movie), rating in ratings.items():
    G.add_edge(user, movie, weight=rating)

# 将节点分为两组
users = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
movies = set(G) - users

# 绘制二部图
pos = nx.bipartite_layout(G, users)
nx.draw_networkx_nodes(G, pos, nodelist=users, node_color='b', label='User')
nx.draw_networkx_nodes(G, pos, nodelist=movies, node_color='r', label='Movie')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): G[u][v]['weight'] for u, v in G.edges()}, label_pos=0.5, font_size=8)
plt.title("User-Movie Interaction Bipartite Graph")
plt.legend()
plt.show()
