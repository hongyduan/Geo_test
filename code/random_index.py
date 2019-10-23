
from torch_geometric.data import Data
import numpy as np
import torch

def random_index_fun(a, delete_num):
    while delete_num!=0:
        ind = np.random.randint(0,a.shape[0],size=1)
        a = a[torch.arange(a.size(0))!=int(ind)]
        time = delete_num -1
    return a


def select_index_fun(G2_edge_index, G2_edge_attr, G2_edge_type, select_num):
    list_ori = np.random.randint(0,G2_edge_index.shape[0],size=select_num)
    list = torch.tensor(list_ori, dtype=torch.long)
    G2_edge_index_new = torch.index_select(
        G2_edge_index,
        dim=0,
        index=list
    )
    G2_edge_attr_new = torch.index_select(
        G2_edge_attr,
        dim=0,
        index=list
    )
    G2_edge_type_new = torch.index_select(
        G2_edge_type,
        dim=0,
        index=list
    )
    return G2_edge_index_new, G2_edge_attr_new, G2_edge_type_new

def get_mini_g2(G2_edge_index, G2_edge_attr, G2_edge_type, select_num, G2_x):
    G2_edge_index_tmp, G2_edge_attr_tmp, G2_edge_type_tmp = select_index_fun(G2_edge_index, G2_edge_attr, G2_edge_type, select_num)
    G2_num_relations_tmp = max(G2_edge_type_tmp.data) + 1
    data_G2_tmp = Data(x=G2_x, edge_index=G2_edge_index_tmp.t().contiguous(), edge_attr=G2_edge_attr_tmp, edge_type=G2_edge_type_tmp, num_relations=G2_num_relations_tmp)
    return data_G2_tmp