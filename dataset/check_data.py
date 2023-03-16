import pickle
import pandas as pd

# cameras 长度 147
cameras = pickle.load(open("../dataset/camera_info.pkl", "rb"))

# node 节点数量 1263
road_graph = pickle.load(open("../dataset/road_graph.pkl", "rb"))


#%%
node_to_A = pickle.load(open("./node_to_A.pkl", 'rb'))
node_to_A_p = pickle.load(open("./node_to_A_p.pkl", 'rb'))

node_to_A[130]
node_to_A_p[130].sum(axis=1)

#%%
"""
camera_node_to_node_to_A
长度： 130
"""
camera_node_to_node_to_A = pickle.load(open("camera_node_to_node_to_A.pkl", 'rb'))
camera_node_to_node_to_A_p = pickle.load(open("camera_node_to_node_to_A_p.pkl", 'rb'))

cid = 475
camera_node_to_node_to_A[cid]

pd.DataFrame(camera_node_to_node_to_A)

