import random
import ast
import csv
import sys
import math
import os

def load_csv_to_header_data(filename,n):
    fs = csv.reader(open(filename))
    all_row = []
    for r in fs:
        all_row.append(r)

    headers = all_row[0]
    idx_to_name, name_to_idx = get_header_name_to_idx_maps(headers)

    data = {
        'header': headers,
        'rows': all_row[1:n],
        'name_to_idx': name_to_idx,
        'idx_to_name': idx_to_name
    }
    return data


def get_header_name_to_idx_maps(headers):
    name_to_idx = {}
    idx_to_name = {}
    for i in range(0, len(headers)):
        name_to_idx[headers[i]] = i
        idx_to_name[i] = headers[i]
    return idx_to_name, name_to_idx


def get_uniq_values(data):
    idx_to_name = data['idx_to_name']
    idxs = idx_to_name.keys()

    val_map = {}
    for idx in iter(idxs):
        val_map[idx_to_name[idx]] = set()

    for data_row in data['rows']:
        for idx in idx_to_name.keys():
            att_name = idx_to_name[idx]
            val = data_row[idx]
            if val not in val_map.keys():
                val_map[att_name].add(val)
    return val_map

def get_class_labels(data, target_attribute):
    rows = data['rows']
    col_idx = data['name_to_idx'][target_attribute]
    labels = {}
    for r in rows:
        val = r[col_idx]
        if val in labels:
            labels[val] = labels[val] + 1
        else:
            labels[val] = 1

    return labels


def entropy(n, labels):
    ent = 0
    for label in labels.keys():
        p_x = labels[label] / n
        ent += - p_x * math.log(p_x, 2)
    return ent


def partition_data(data, group_att):
    partitions = {}
    data_rows = data['rows']
    partition_att_idx = data['name_to_idx'][group_att]
    for row in data_rows:
        row_val = row[partition_att_idx]
        if row_val not in partitions.keys():
            partitions[row_val] = {
                'name_to_idx': data['name_to_idx'],
                'idx_to_name': data['idx_to_name'],
                'rows': list()
            }
        partitions[row_val]['rows'].append(row)
    return partitions


def avg_entropy_in_partitions(data, splitting_att, target_attribute):
    # znajdz unikatowe wartości w split
    data_rows = data['rows']
    n = len(data_rows)
    partitions = partition_data(data, splitting_att)

    avg_ent = 0

    for partition_key in partitions.keys():
        partitioned_data = partitions[partition_key]
        partition_n = len(partitioned_data['rows'])
        partition_labels = get_class_labels(partitioned_data, target_attribute)
        partition_entropy = entropy(partition_n, partition_labels)
        avg_ent += partition_n / n * partition_entropy

    return avg_ent, partitions

def roulett_attribute(info_gains_for_roulete,info_gains_partition_name):
    max_info_gain = None
    max_info_gain_att = None
    max_info_gain_partitions = None
    roulete_answer = 0
    gains = []
    distribution_function = []
    buff = 0
    buff_temp = 0

    for inf_gain in info_gains_for_roulete:
        g = inf_gain/sum(info_gains_for_roulete)
        gains.append(g)

    for g in gains:
        buff = g + buff_temp
        distribution_function.append(buff)
        buff_temp = buff

    rnd = random.uniform(0, 1)

    for dist in distribution_function:
        if dist > rnd:
            continue
        if dist < rnd:
            roulete_answer = distribution_function.index(dist)

    return info_gains_for_roulete[roulete_answer] , info_gains_partition_name[roulete_answer]

def id3(data, uniqs, remaining_atts, target_attribute,mode,depth):

    labels = get_class_labels(data, target_attribute)
    node = {}
    node['depth'] = depth
    # print(node['depth'])
    # kryterium stopu
    if len(labels.keys()) == 1:
        node['label'] = next(iter(labels.keys()))
        return node

    n = len(data['rows'])
    ent = entropy(n, labels)

    max_info_gain = None
    max_info_gain_att = None
    max_info_gain_partitions = None
    info_gains_for_roulete = []
    info_gains_partition_name = []

    # wybór atrybutu do wykonania split
    for remaining_att in remaining_atts:
        avg_ent, partitions = avg_entropy_in_partitions(data, remaining_att, target_attribute)
        info_gain = ent - avg_ent
        info_gains_for_roulete.append(info_gain)
        info_gains_partition_name.append([remaining_att,partitions])
        if mode == 'simple':
            if max_info_gain is None or info_gain > max_info_gain:
                max_info_gain = info_gain
                max_info_gain_att = remaining_att
                max_info_gain_partitions = partitions

    if mode == 'roulete':
        max_info_gain, inf_gain_inf =  roulett_attribute(info_gains_for_roulete,info_gains_partition_name)
        max_info_gain_att = inf_gain_inf[0]
        max_info_gain_partitions = inf_gain_inf[1]


    node['attribute'] = max_info_gain_att
    node['nodes'] = {}
    uniq_att_values = uniqs[max_info_gain_att]

    #split oraz dodanie węzła "dziecko"
    for att_value in uniq_att_values:
        if att_value not in max_info_gain_partitions.keys():
            continue
        partition = max_info_gain_partitions[att_value]
        node['nodes'][att_value] = id3(partition, uniqs, remaining_atts, target_attribute,mode,depth+1)


    return node

def class_counts(rows):
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        label = row[0]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts



def classify(root,item,headers):
    answer = set()
    def check(node):
        if 'label' in node:
            answer.add(node['label'])
        elif 'attribute' in node:
            for h in headers:
                if node['attribute'] == h:
                    att_idx = headers.index(h)
                    break
            for subnode_key in node['nodes']:
                if subnode_key == item[att_idx]:
                    check(node['nodes'][subnode_key])
    check(root)
    if not answer:
        answer.add('b')
    return answer

def accuracy(tree,test_data):
    items = test_data['rows']
    item_headers = test_data['header']
    item_headers.pop(0)
    count_good_answers = 0
    for item in items:
        item_value = item[0]
        item.pop(0)
        ans = classify(tree,item,item_headers)

        if next(iter(ans)) == item_value:
             count_good_answers +=1
        else:
            continue
    return count_good_answers/len(items)

def cut_tree(tree,label_value,attribute_name):
    new_tree ={}
    cut = False
    def check(node,new_tree,cut):
        if 'label' in node:
            if cut == True:
                new_tree['label'] = label_value
                cut = False
        elif 'attribute' in node:
            new_tree['attribute'] = node['attribute']
            new_tree['nodes'] = {}
            if node['attribute'] == attribute_name:
                cut = True
            for subnode_key in node['nodes']:
                new_tree['nodes'][subnode_key] = node['nodes'][subnode_key]
                check(node['nodes'][subnode_key],new_tree,cut)

    check(tree,new_tree,cut)

    return new_tree

def pruning(tree,test_data):
    value  = {}
    list_values =[]
    def check(node,value,valueAttr,list_values):
        if 'label' in node:
            value['label'] = node['label']
            list_values.append(value)
        elif 'attribute' in node:
            for subnode_key in node['nodes']:
                value = {}
                value['attr'] = node['attribute']
                value['valAttr'] = subnode_key
                check(node['nodes'][subnode_key],value,subnode_key,list_values)
    check(tree,value,None,list_values)

    pos_neg_list = []

    def classik(tree,pos_neg_list):
        if 'attribute' in tree:
            for subnode_key in tree['nodes']:
                count_neg = 0
                count_pos = 0
                attr = tree['attribute']
                for i in range(len(list_values)):
                    if attr == list_values[i]['attr']:
                        if list_values[i]['label'] == 'e':
                            count_pos +=1
                        else:
                            count_neg +=1
                pos_neg_list.append([tree['attribute'],count_pos,count_neg])
                classik(tree['nodes'][subnode_key],pos_neg_list)

    classik(tree,pos_neg_list)

    pos_neg_list_unique = []
    for x in pos_neg_list:
        if x not in pos_neg_list_unique:
            pos_neg_list_unique.append(x)

    for i in range(len(pos_neg_list_unique)):
        max_val = max(pos_neg_list_unique[i][1],pos_neg_list_unique[i][2])
        min_val =  min(pos_neg_list_unique[i][1],pos_neg_list_unique[i][2])
        lis = pos_neg_list_unique[i]
        index_max = lis.index(max_val)
        index_min =  lis.index(min_val)
        diff = max_val - min_val
        if diff >= 3:
            if lis[index_max] == 1:
                label_value = 'e'
            else:
                label_value = 'p'
            attribute_name = lis[0]
            new_tree = cut_tree(tree,label_value,attribute_name)
            val1 = accuracy(new_tree,test_data)
            val2 =  accuracy(tree,test_data)-0.1
            if val1 <= val2:
                continue
            else:
                tree = new_tree

    print(tree)

def main():
    mode ='roulete'
    #mode ='simple'
    argv = sys.argv
    data = load_csv_to_header_data(argv[1],6500)
    target_attribute = data['header'][0]
    remaining_attributes = set(data['header'])
    remaining_attributes.remove(target_attribute)
    uniqs = get_uniq_values(data)

    root = id3(data, uniqs, remaining_attributes, target_attribute,mode,0)

    test_data = load_csv_to_header_data(argv[2],1400)


    print(accuracy(root,test_data))
    # pruning(root,test_data)

    print(root)


if __name__ == "__main__": main()
