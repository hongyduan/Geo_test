from layer_specific_to_common import STCConv
from torch_geometric.data import Data
from random_index import *
from load_data import *
from pre_data import *
from net import *
import argparse
import torch



# x: has shape [N, in_channels]
# edge_index: has shape [2, E]
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--type_path', type=str, default="/Users/bubuying/PycharmProjects/Geometric_test/data/type_embedding/node_embedding.npy")
    parser.add_argument('--entity_path', type=str, default="/Users/bubuying/PycharmProjects/Geometric_test/data/entity_embedding/node_embedding.npy")
    parser.add_argument('--entity_relation_path', type=str, default="/Users/bubuying/PycharmProjects/Geometric_test/data/entity_embedding/node_re_embedding.npy")
    parser.add_argument('--data_path', type=str, default="/Users/bubuying/PycharmProjects/Geometric_test/data/yago_result")
    parser.add_argument('--data_path_bef', type=str, default="/Users/bubuying/PycharmProjects/Geometric_test/data/yago")
    parser.add_argument('--G2_val_file_name', type=str, default="val_entity_Graph.txt")
    parser.add_argument('--G2_test_file_name', type=str, default="test_entity_Graph.txt")

    parser.add_argument('--dim', type=int, default=500)
    parser.add_argument('--in_dim', type=int, default=500)
    parser.add_argument('--out_dim', type=int, default=500)
    parser.add_argument('--class_num', type=int, default=106)
    parser.add_argument('--class_num_double', type=int, default=200)
    parser.add_argument('--select_num', type=int, default=70000)
    parser.add_argument('--leaf_node_entity', action='store_true', default=True)

    return parser.parse_args(args)

def pre_load_data(args):

    # pre_data
    # all_node_embedding_: 26989*500;       # G1_node_embedding_: [0]:911*500     [1]:106*500    [2]:9962*500       # G2_node_embedding_: 26078*500;
    all_node_embedding_ = all_node_embedding(args.entity_path, args.type_path)

    # load_data
    # node2id_G2_re: 26078;   relation2id_G2_re: 34;    type_node2id_G1_re: 911;   G1_graph_sub3: 106;     G1_graph_sub2: 8948;     G1_graph_sub1: 894;     G2_links: 228619
    node2id_G2_re, relation2id_G2_re, type_node2id_G1_re, G1_graph_sub3, G1_graph_sub2, G1_graph_sub1, val_data, test_data = load_da(args.data_path, args.data_path_bef)

    ty_index_G3 = list(G1_graph_sub3.keys())
    ty_index_G3_dict = dict()
    ind=0
    for i in ty_index_G3:
        ty_index_G3_dict[i] = int(ind)
        ind = ind+1
    G1_graph_sub2_new = OrderedDict()
    for key, values in G1_graph_sub2.items():
        if key not in G1_graph_sub2_new.keys():
            G1_graph_sub2_new[key] = list()
        for value in values:
            if ty_index_G3_dict[value] not in G1_graph_sub2_new[key]:
                G1_graph_sub2_new[key].append(int(ty_index_G3_dict[value]))


    en_index_G3 = list(G1_graph_sub2.keys())

    en_index_G3_list = list()
    for i in en_index_G3:
        en_index_G3_list.append(int(i))
    en_index_G3_list_train_bef = en_index_G3_list[0:6178]
    en_index_G3_list_test_bef = en_index_G3_list[6178:len(en_index_G3_list)]

    en_index_G3_list = torch.tensor(en_index_G3_list, dtype=torch.long) # 6178+1544 entity
    en_index_G3_list_train = torch.tensor(en_index_G3_list_train_bef, dtype=torch.long)   # 6178 entity for train
    en_index_G3_list_test = torch.tensor(en_index_G3_list_test_bef, dtype=torch.long)   # 1544 entity for test


    en_embedding_G3 = torch.index_select(    # (6178+1544)*500 embedding
        all_node_embedding_,
        dim=0,
        index=en_index_G3_list
    )

    en_embedding_G3_train = torch.index_select(   # 6178*500 embedding
        all_node_embedding_,
        dim=0,
        index=en_index_G3_list_train
    )

    en_embedding_G3_test = torch.index_select(   # 1544*500 embedding
        all_node_embedding_,
        dim=0,
        index=en_index_G3_list_test
    )

    G1_node_embedding_ = G1_node_embedding(args.type_path, all_node_embedding_, G1_graph_sub3)
    G1_node_embedding_type_, G1_node_embedding_type_small_, G1_ndoe_embedding_entity_ = G1_node_embedding_
    G2_node_embedding_ = G2_node_embedding(args.entity_path)


    # load train data: G1
    # x:[N, in_channels];    edge_index:[2,E];    G1_graph_sub2(en is_instance_of ty)  G1_graph_sub1(ty1 is_a ty2)
    edge_index_G1_, edge_index_G1_sub2_, edge_index_G1_sub1_ = edge_index_G1(G1_graph_sub2, G1_graph_sub1)
    if args.leaf_node_entity:
        # G1_x = torch.cat((G1_node_embedding_type_, G1_ndoe_embedding_entity_), 0)  # (911+8948)*500
        G1_x = all_node_embedding_
        G1_edge_index = edge_index_G1_ # (8962+8467) edges
    else:
        G1_x = G1_node_embedding_type_  # 911*500
        G1_edge_index = edge_index_G1_sub1_  # 8962 edges
    data_G1 = Data(x = G1_x, edge_index = G1_edge_index.t().contiguous())
    # load val data: G1     # load test data: G1
    G1_edge_index_val, G1_edge_index_test = edge_index_G1_val_test(val_data[0],test_data[0])
    data_G1_val = Data(edge_index = G1_edge_index_val.t().contiguous())
    data_G1_test = Data(edge_index = G1_edge_index_test.t().contiguous())



    # load train data: G2；   G2_graph (en1 relaiton en2) undirected
    G2_x = G2_node_embedding_  # 26078*500
    G2_edge_index, G2_edge_attr, G2_edge_type, G2_num_relations = edge_index_attr_G2(args.data_path, args.entity_relation_path, node2id_G2_re, relation2id_G2_re, args.dim)  # 332127 edges
    data_G2 = Data(x = G2_x, edge_index = G2_edge_index.t().contiguous(), edge_attr = G2_edge_attr, edge_type = G2_edge_type, num_relations = G2_num_relations)

    # load val data: G2
    G2_edge_index_val, G2_edge_attr_val = edge_index_attr_G2_val_test(args.data_path,args.entity_relation_path,node2id_G2_re, relation2id_G2_re, args.dim, args.G2_val_file_name)
    data_G2_val = Data(edge_index = G2_edge_index_val.t().contiguous(), edge_attr = G2_edge_attr_val)

    # load test data: G2
    G2_edge_index_test, G2_edge_attr_test = edge_index_attr_G2_val_test(args.data_path,args.entity_relation_path,node2id_G2_re, relation2id_G2_re, args.dim, args.G2_test_file_name)
    data_G2_test = Data(edge_index = G2_edge_index_test.t().contiguous(), edge_attr = G2_edge_attr_test)


    return G1_graph_sub2_new, en_index_G3_list_train_bef, en_index_G3_list_test_bef, en_index_G3_list_train, en_index_G3_list_test, G2_edge_index, all_node_embedding_, G1_node_embedding_, G2_node_embedding_, data_G1, data_G2, data_G1_val, data_G1_test, data_G2_val, data_G2_test, en_embedding_G3, en_embedding_G3_train, en_embedding_G3_test


def loss(out):

    return 1

def score():
    return 1


def main(args):

    # all_node_embedding_: 26989*500;  # G1_node_embedding_: [0]:911*500     [1]:106*500    [2]:8948*500 # G2_node_embedding_: 26078*500;
    G1_graph_sub2_new, en_index_G3_list_train_bef, en_index_G3_list_test_bef, en_index_G3_list_train, en_index_G3_list_test, G2_edge_index_, all_node_embedding_, G1_node_embedding_, G2_node_embedding_, data_G1, data_G2, data_G1_val, data_G1_test, data_G2_val, data_G2_test, en_embedding_G3,en_embedding_G3_train,en_embedding_G3_test = pre_load_data(args)

    print("... ...for debug enter main funciton")

    # device = torch.device('cuda' if torch.cuda.is_available() else'cpu')

    model = Net(args, data_G2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)



    model.train()
    for epoch in range(20):
        for batch in range(10):
            print("... ...for debug beginning a batch")
            optimizer.zero_grad()
            data_G2_mini = get_mini_g2(G2_edge_index_, data_G2.edge_attr, data_G2.edge_type, args.select_num, data_G2.x)
            out_G3 = model(data_G1, data_G2_mini)
            train_subset_of_out_G3_train = torch.index_select(  # 6178*500 embedding
                out_G3,
                dim=0,
                index=en_index_G3_list_train
            )

            target_tmp = torch.zeros((train_subset_of_out_G3_train.shape[0],train_subset_of_out_G3_train.shape[1]),dtype=torch.float32)
            i=0
            for inn in en_index_G3_list_train_bef:
                for value in G1_graph_sub2_new[str(inn)]:
                    target_tmp[i,int(value)] = 1
                i = i +1

            target = target_tmp
            loss = F.binary_cross_entropy(train_subset_of_out_G3_train, target)
            print('loss:{}'.format(loss))
            loss.backward()  # 误差反向传播计算参数梯度
            optimizer.step()  # 通过梯度 做参数更新
            print("... ...for debug finished a batch")

    # test
    print("... ...for debug beginning test")
    model.eval()
    out_test = model(data_G1, data_G2)
    train_subset_of_out_G3_test = torch.index_select(  # 1544*106 embedding
        out_test,
        dim=0,
        index=en_index_G3_list_test
    )
    target_tmp = torch.zeros((train_subset_of_out_G3_test.shape[0], train_subset_of_out_G3_test.shape[1]),
                             dtype=torch.float32)

    i = 0
    for inn in en_index_G3_list_test_bef:
        for value in G1_graph_sub2_new[str(inn)]:
            target_tmp[i, int(value)] = 1
        i = i + 1

    target = target_tmp  # 1544*106
    all_acc = 0
    for line in train_subset_of_out_G3_test.shape[0]:
        tmp1 = train_subset_of_out_G3_test[line]
        tmp2 = target[line]
        acc = tmp1.eq(tmp2).sum().item() / tmp1.shape[0]
        all_acc = all_acc + acc
    final_acc = all_acc/train_subset_of_out_G3_test.shape[0]

    print(final_acc)
    print("... ...for debug finished test")

if __name__ == '__main__':
    main(parse_args())
