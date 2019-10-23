import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.nn import RGCNConv

name = 'MUTAG'  # MUTAG: nodes,23644  relations,23  edges,74227  labeled,340  classes,2
path = osp.join(
    osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities', name)
dataset = Entities(path, name)
data = dataset[0]

print(data)
print(data.num_nodes)
print(dataset.num_relations)
print(dataset.num_classes)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = RGCNConv(
            data.num_nodes, 16, dataset.num_relations, num_bases=30)  # 23644, 16, 46, 30
        self.conv2 = RGCNConv(
            16, dataset.num_classes, dataset.num_relations, num_bases=30)  # 16, 2, 46, 30

    def forward(self, edge_index, edge_type, edge_norm):
        print(edge_index.size())  # [2,148454]
        print(edge_type.size())  # [148454]
        print(edge_norm.size())  # [148454]
        x = F.relu(self.conv1(None, edge_index, edge_type))  #((None), (2, 148454), (148454))
        print(x.size())  # [23644,16]
        x = self.conv2(x, edge_index, edge_type)  # ((23644,16), (2, 148454), (148454))
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)


def train():
    model.train()
    optimizer.zero_grad()
    print(data.edge_index)
    print(data.edge_type)
    print(data.edge_norm)
    out = model(data.edge_index, data.edge_type, data.edge_norm)
    F.nll_loss(out[data.train_idx], data.train_y).backward()
    optimizer.step()


def test():
    model.eval()
    out = model(data.edge_index, data.edge_type, data.edge_norm)
    pred = out[data.test_idx].max(1)[1]
    acc = pred.eq(data.test_y).sum().item() / data.test_y.size(0)
    return acc


for epoch in range(1, 51):
    train()
    test_acc = test()
    print('Epoch: {:02d}, Accuracy: {:.4f}'.format(epoch, test_acc))