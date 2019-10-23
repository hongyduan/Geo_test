

import numpy as np
import torch
import time
import os


def all_node_embedding(entity_path, type_path):
    node_embedding_entity = np.load(entity_path)
    node_embedding_type = np.load(type_path)
    all_node_embedding = torch.from_numpy(np.vstack((node_embedding_entity, node_embedding_type)))
    return all_node_embedding


def G1_node_embedding(type_path, all_node_embedding, G1_graph_sub3):
    # G1_node_embedding_type:  906 type embedding
    G1_node_embedding_type = torch.from_numpy(np.load(type_path))

    # G1_node_embedding_type_small:  106 type embedding
    # G1_ndoe_embedding_entity:  8948 entity embedding
    left_common = list()  #106
    right_specific = list()  # 8948
    for left_comm, right_speci in G1_graph_sub3.items():
        if left_comm not in left_common:
            left_common.append(left_comm)
        for values in right_speci:
            values = int(values)
            if values not in right_specific:
                right_specific.append(values)
    G1_node_embedding_type_small = all_node_embedding[left_common]
    G1_ndoe_embedding_entity = all_node_embedding[right_specific]
    return G1_node_embedding_type, G1_node_embedding_type_small, G1_ndoe_embedding_entity


def G2_node_embedding(entity_path):
    # 26078 entity embedding
    G2_node_embedding_entity = torch.from_numpy(np.load(entity_path))
    return G2_node_embedding_entity

def edge_index_G1(G1_graph_sub2, G1_graph_sub1):
    # G1_graph_sub2(en is_instance_of ty)  G1_graph_sub1(ty1 is_a ty2)
    templist1 = list()
    for source, targets in G1_graph_sub2.items():
        for target in targets:
            templist1.append([int(source), int(target)])
    edge_index_G1_sub2 = torch.tensor(templist1, dtype=torch.long)  # edges: 9962
    templist2 = list()
    for source, targets in G1_graph_sub1.items():
        for target in targets:
            templist2.append([int(source), int(target)])
            templist1.append([int(source), int(target)])
    edge_index_G1_sub1 = torch.tensor(templist2, dtype=torch.long) # edges: 8962
    edge_index_G1 = torch.tensor(templist1, dtype=torch.long)
    return edge_index_G1, edge_index_G1_sub2, edge_index_G1_sub1


def edge_index_G1_val_test(G1_val, G1_test):
    # G1_val: en_ty/499; G1_test: en_ty/996
    templist = list()
    for source, targets in G1_val.items():
        for target in targets:
            templist.append([int(source), int(target)])
    G1_edge_index_val = torch.tensor(templist, dtype=torch.long)
    templist = list()
    for source, targets in G1_test.items():
        for target in targets:
            templist.append([int(source), int(target)])
    G1_edge_index_test = torch.tensor(templist, dtype=torch.long)
    return G1_edge_index_val, G1_edge_index_test


def edge_index_attr_G2(data_path, entity_relation_path, node2id_G2_re, relation2id_G2_re, dim):
    # G2_graph(en1 relation en2)
    f = open(os.path.join(data_path, 'train_entity_Graph.txt'), "r")
    num_lines = len(f.readlines())*2
    i = 0
    templist_index = [None] * num_lines
    edge_attr_G2 = torch.empty(num_lines, dim)
    edge_type_G2 = torch.empty(num_lines, dtype=torch.long)
    num_relations = list()
    G2_relation_embedding = np.load(entity_relation_path)  # 34*500
    with open(os.path.join(data_path, 'train_entity_Graph.txt')) as fin:
        for line in fin:
            en1, r, en2 = line.strip().split('\t')
            en1id = node2id_G2_re[en1]
            rid = relation2id_G2_re[r]
            en2id = node2id_G2_re[en2]
            if rid not in num_relations:
                num_relations.append(rid)
            templist_index[i] = [int(en1id),int(en2id)]
            edge_attr_G2[i] = torch.from_numpy(G2_relation_embedding[int(rid)]).unsqueeze(0)
            edge_type_G2[i] = int(rid)
            i = i+1
            templist_index[i] = [int(en2id), int(en2id)]
            edge_attr_G2[i] = torch.rand(1,500)
            edge_type_G2[i] = int(rid)
            i = i+1

    num_relations = len(num_relations)
    edge_index_G2 = torch.tensor(templist_index, dtype=torch.long) # edges: 2*332127
    edge_attr_G2 = edge_attr_G2  # attr: (2*332127)*500
    return edge_index_G2, edge_attr_G2, edge_type_G2, num_relations


def edge_index_attr_G2_val_test(data_path, entity_relation_path, node2id_G2_re, relation2id_G2_re, dim, file_name):
    f = open(os.path.join(data_path, file_name), 'r')
    num_lines = len(f.readlines())
    i = 0
    templist_index = [None] * num_lines
    edge_attr_G2_val_test = torch.empty(num_lines, dim)
    G2_relation_embedding = np.load(entity_relation_path)
    with open(os.path.join(data_path, file_name)) as fin:
        for line in fin:
            en1, r, en2 = line.strip().split('\t')
            en1id = node2id_G2_re[en1]
            rid = relation2id_G2_re[r]
            en2id = node2id_G2_re[en2]

            templist_index[i] = [int(en1id), int(en2id)]
            edge_attr_G2_val_test[i] = torch.from_numpy(G2_relation_embedding[int(rid)]).unsqueeze(0)
            i = i+1
    edge_index_G2_val_test = torch.tensor(templist_index, dtype=torch.long) # edges: 19538
    edge_attr_G2_val_test = edge_attr_G2_val_test # attr: 19538*500
    return edge_index_G2_val_test, edge_attr_G2_val_test
















