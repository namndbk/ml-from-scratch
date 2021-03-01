import numpy as np
import pandas as pd
from itertools import combinations


def parse(file_name):
    # doc data bang pandas
    data = pd.read_csv(file_name)
    # bien X luu cac transaction
    X = []
    # items luu danh sach item
    items = {}
    # count - stt cua item
    count = 0
    for i in range(data.shape[0]):
        # bien x luu cac item co trong mot transaction
        x = []
        for j in range(data.shape[1]):
            # lay item thu j cua transaction i
            item = data.iloc[i, j]
            # neu nan hay null thi bo qua
            if str(item) == "nan":
                continue
            else:
                # neu item do co trong danh thi them vao transaction x
                if item.lower().strip() in items:
                    x.append(items[item.lower().strip()])
                # neu item do khong co trong danh sach items thi them vao ds items va them vao transaction x
                else:
                    items[item.lower().strip()] = count
                    x.append(count)
                    count += 1
        X.append(x)
    return X, items


def gen_candidate(X):
    c1 = []
    # duyet qua tung transaction, tao ra cac item set co kich thuoc la 1
    for transaction in X:
        for i in transaction:
            i = frozenset([i])
            if i not in c1:
                c1.append(i)
    return c1


def gen_candidate_k(freq_item, k):
    """
    tao danh sach gom cack-itemset
    """
    ck = []

    # tao ung vien co kich thuoc la 2 (2-itemset)
    if k == 0:
        for f1, f2 in combinations(freq_item, 2):
            item = f1 | f2  # ket hop 2 item set
            ck.append(item)
    else:
        for f1, f2 in combinations(freq_item, 2):
            # neu 2 k+1 (item set) co k phan tu chung thi chung se hop nhat thanh ung cu vien k+2 itemset
            intersection = f1 & f2
            if len(intersection) == k:
                item = f1 | f2
                if item not in ck:
                    ck.append(item)
    return ck


def create_freq_item(X, ck, min_support):
    """
    tim ra cac itemset co support >= min_support
    """
    # lap lai tung transaction va tinh toan so luong tung item co xuat hien trong du lieu
    item_count = {}
    for transaction in X:
        for item in ck:
            if item.issubset(transaction):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1

    n_row = X.shape[0]
    #freq_item tan suat xuat hien cua tung item set, luu danh sach item set xuat hien thuong xuyen
    freq_item = []
    # do support cua tung item set
    item_support = {}

    # tinh do support, neu >= min_support thi them no vao freq_item
    for item in item_count:
        support = item_count[item] / n_row
        if support >= min_support:
            freq_item.append(item)

        item_support[item] = support

    return freq_item, item_support


def apriori(X, min_support):
    #gen 1-itemset
    c1 = gen_candidate(X)
    # gen freq_item tu 1-item
    freq_item, item_support_dict = create_freq_item(X, c1, min_support=0.01)
    # freq_items danh sach k-itemset
    freq_items = [freq_item]

    #duyet cac item set pho bien
    k = 0
    while len(freq_items[k]) > 0:
        #lay ra danh sach k - item set
        freq_item = freq_items[k]
        # gen ra k+2 itemset 
        ck = gen_candidate_k(freq_item, k)
        #tim cac k-itemset pho bien
        freq_item, item_support = create_freq_item(X, ck, min_support=0.01)
        freq_items.append(freq_item)
        item_support_dict.update(item_support)
        k += 1

    return freq_items, item_support_dict


def compute_conf(freq_items, item_support_dict, freq_set, subsets, min_confidence):
    """
    sinh luat va check min_confidence
    """
    rules = []
    right_hand_side = []

    for rhs in subsets:
        # tao danh sach ben trai cua luat lien ket, them vao rules neu luat sinh ra > min_confidence
        #lhs - danh sach item phia trai cua luat lien ket, rhs - danh sach item phia ben phai
        lhs = freq_set - rhs
        #tinh confidence cua luat
        conf = item_support_dict[freq_set] / item_support_dict[lhs]
        #check neu luat du dieu kien >= min_conf
        if conf >= min_confidence:
            lift = conf / item_support_dict[rhs]
            #rules-luat lien ket
            rules_info = lhs, rhs, conf, lift
            rules.append(rules_info)
            right_hand_side.append(rhs)

    return rules, right_hand_side


def create_rules(freq_items, item_support_dict, min_confidence):
    association_rules = []

    #duyet qua tung danh sach cac k-item set pho bien
    #tinh tu k = 2, do k =1 itemset chi co mot phan tu thi khong the tach thanh lhs va rhs
    for idx, freq_item in enumerate(freq_items[1:(len(freq_items) - 1)]):
        for freq_set in freq_item:

            subsets = [frozenset([item]) for item in freq_set]
            rules, right_hand_side = compute_conf(
                freq_items, item_support_dict, freq_set, subsets, min_confidence)
            association_rules.extend(rules)

            #duyet qua cac itemset pho bien, size itemset tu k=3 tro len
            if idx != 0:
                k = 0
                while len(right_hand_side[0]) < len(freq_set) - 1:
                    ck = gen_candidate_k(right_hand_side, k=k)
                    rules, right_hand_side = compute_conf(freq_items, item_support_dict,
                                                          freq_set, ck, min_confidence)
                    association_rules.extend(rules)
                    k += 1

    return association_rules


if __name__ == "__main__":
	X, items = parse("data/a1_market_basket_optimisation.csv")
	X = np.array(X)
	freq_items, item_support_dict = apriori(X, min_support = 0.01)
	association_rules = create_rules(freq_items, item_support_dict, min_confidence=0.5)