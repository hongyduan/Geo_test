import torch
from layer_specific_to_common import *
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import RGCNConv

class Net(nn.Module):
    def __init__(self, args, data_G2):
        super(Net, self).__init__()

        self.x_g2 = nn.Embedding(data_G2.x.shape[0], 200)   # 26078*500

        self.layer_g3_rgcn_1 = RGCNConv(data_G2.num_nodes, args.class_num_double, data_G2.num_relations, num_bases=30)
        self.layer_g3_rgcn_2 = RGCNConv(args.class_num_double, args.class_num, data_G2.num_relations, num_bases=30)

        print("for debug")

    def forward(self, data_G1, data_G2):

        em_check = torch.mean((self.x_g2.weight.data) ** 2)
        print('em_mean_before_net_forward:{}'.format(em_check))

        en_em_g3 = self.layer_g3_rgcn_1(self.x_g2.weight.data, data_G2.edge_index, data_G2.edge_type, None, [data_G2.num_nodes, data_G2.num_nodes])
        en_em_g3 = F.relu(en_em_g3)
        en_em_g3 = self.layer_g3_rgcn_2(en_em_g3, data_G2.edge_index, data_G2.edge_type)

        em_check = torch.mean((self.x_g2.weight.data) ** 2)
        print('em_mean_after_net_forward:{}'.format(em_check))
        return F.softmax(en_em_g3, dim=1)




