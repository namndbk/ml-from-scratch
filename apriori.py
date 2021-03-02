import numpy as np
import pandas as pd
from itertools import combinations


def parse(file_name: str):
    """
    Convert input CSV data to a tuple of transaction list and dictionary of transaction items:

    @param file_name: location of CSV data file
    @type file_name: str
    @return: transaction list: item words are converted to sequential index
             transaction items: a dictionary mapping item word with its index
    @rtype: Tuple[List[Set[int]], dict]
    """

    # read csv file with pandas
    data = pd.read_csv(file_name)

    # Variable transaction_list - to store transaction
    transaction_list = []

    # Use a dict to store items from the given transactions
    items_dict = {}

    # index_count - sequential index of each item
    index_count = 0

    # total number of transactions of the given data
    total_transaction_number = data.shape[0]

    # maximum number of items per transaction (max number of columns) of the given data
    max_items_per_tranx = data.shape[1]

    for i in range(total_transaction_number):

        # items_each_trx to store items in each transaction
        items_each_trx = []

        for j in range(max_items_per_tranx):

            # get item j (position) of transaction i
            item = data.iloc[i, j]
            # if item is NaN or Null -> skip the loop
            if str(item) == "nan":
                continue
            else:
                # if the item is in items_dict, to add its index to transaction_list
                if item.lower().strip() in items_dict:
                    items_each_trx.append(items_dict[item.lower().strip()])

                # if the item is NOT in items_dict, to compute its index, add to items_dict and
                # add its index to transaction_list
                else:
                    items_dict[item.lower().strip()] = index_count
                    items_each_trx.append(index_count)
                    index_count += 1
        transaction_list.append(items_each_trx)

    return transaction_list, items_dict


def get_candidates(transaction_list: list):
    """
    Generate size-one-candidates (one-item sets) from the list of transactions
    @param transaction_list: list of transactions with transaction items being coded by their sequential indexes
    @type transaction_list: List[Set[int]]
    @return: list of sets of size-one-candidate (one-item sets)
    @rtype: List[set]
    """

    # size_one_candidates: a list to store size-one candidates (one-item sets)    
    size_one_candidates = []
    # loop through each transaction to create  item sets of size-one (one-item sets)
    for transaction in transaction_list:
        for i in transaction:
            i = frozenset([i])
            if i not in size_one_candidates:
                size_one_candidates.append(i)
    return size_one_candidates


def generate_candidate_k(freq_item: list, k: int) -> list:
    """
    Generate list of  of size-k candidates from the  frequent item list (freq_item)

    @param freq_item: list of qualified item sets (item sets with support >= min_support
    @type freq_item:  list
    @param k:
    @type k:
    @return:
    @rtype:
    """
    size_k_candidates = []

    # to generate candidates of size two (2-item sets)

    if k == 0:
        # TODO: giải thích tại sao?
        for f1, f2 in combinations(freq_item, 2):
            item = f1 | f2  # union of two sets)
            size_k_candidates.append(item)

    # k > 0
    else:
        for f1, f2 in combinations(freq_item, 2):
            # if the two (k+1)-item sets both has k common items
            # then they will be combined to be the (k+2)-item candidate
            # TODO: giải thích tại sao?
            intersection = f1 & f2
            if len(intersection) == k:
                item = f1 | f2
                if item not in size_k_candidates:
                    size_k_candidates.append(item)

    return size_k_candidates


def create_freq_item(transaction_list, size_k_candidates, min_support):
    """
    find candidates whose support >= min_support
    @param transaction_list: list of transactions with transaction items being coded by their sequential indexes
    @type transaction_list: List[Set[int]]
    @param size_k_candidates:
    @type size_k_candidates:
    @param min_support: The minimum fraction of transactions an itemets needs to occur in to be deemed frequent
    @type min_support: float
    @return:                    
    @rtype:

    """
    # loop through the transaction and compute the count for each candidate (item)
    item_count = {}

    # check in each transaction
    for transaction in transaction_list:

        # to see if the items of each size-k candidate is part (subset) of the transaction
        # if the candidate is the subset add to item_count dictionary together with its count
        for candidate in size_k_candidates:
            if candidate.issubset(transaction):
                if candidate not in item_count:
                    item_count[candidate] = 1
                else:
                    item_count[candidate] += 1

    # total number of transactions
    n_row = transaction_list.shape[0]

    # freq_item indicates the frequency of occurence of each item set. To store the frequent item sets
    freq_item = []

    # if the support of an item is greater than the min_support, then it is considered as frequent
    # compute support of each item set , if >= min_support -> append that item set to freq_item
    item_support = {}

    for item in item_count:
        support = item_count[item] / n_row
        if support >= min_support:
            freq_item.append(item)

        item_support[item] = support

    return freq_item, item_support


def apriori(transaction_list, min_support=0.01):
    """
    @param transaction_list:
    @type transaction_list:
    @param min_support:
    @type min_support:
    @return:
    @rtype:
    """
    # generate size-one-candidates (one-item sets) from the list of transactions
    size_one_candidates = get_candidates(transaction_list)

    # find candidates/item sets whose support >= min_support from size_one_candidates
    freq_item, item_support_dict = create_freq_item(transaction_list, size_one_candidates, min_support=min_support)

    # generate size-k candidates from the list of qualified qualified size-one item sets (aka freq_items)
    freq_items = [freq_item]

    # loop through freq_items and add list of k-incremental-size to it while looping
    k = 0
    while len(freq_items[k]) > 0:

        # get list of size-k item sets
        freq_item = freq_items[k]
        
        # generate k+2 itemset
        size_k_candidates = generate_candidate_k(freq_item, k)

        # find frequent item sets/candidates of k-size
        freq_item, item_support = create_freq_item(transaction_list, size_k_candidates, min_support=min_support)

        # append to the qualified frequent item set to freq_items and update their support
        freq_items.append(freq_item)
        item_support_dict.update(item_support)

        # increment k and keep looping to add until exhausted
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
        # lhs - danh sach item phia trai cua luat lien ket, rhs - danh sach item phia ben phai
        lhs = freq_set - rhs
        # tinh confidence cua luat
        conf = item_support_dict[freq_set] / item_support_dict[lhs]
        # check neu luat du dieu kien >= min_conf
        if conf >= min_confidence:
            lift = conf / item_support_dict[rhs]
            # rules-luat lien ket
            rules_info = lhs, rhs, conf, lift
            rules.append(rules_info)
            right_hand_side.append(rhs)

    return rules, right_hand_side


def create_rules(freq_items, item_support_dict, min_confidence):
    association_rules = []

    # duyet qua tung danh sach cac k-item set pho bien
    # tinh tu k = 2, do k =1 itemset chi co mot phan tu thi khong the tach thanh lhs va rhs
    for idx, freq_item in enumerate(freq_items[1:(len(freq_items) - 1)]):
        for freq_set in freq_item:

            subsets = [frozenset([item]) for item in freq_set]
            rules, right_hand_side = compute_conf(
                freq_items, item_support_dict, freq_set, subsets, min_confidence)
            association_rules.extend(rules)

            # duyet qua cac itemset pho bien, size itemset tu k=3 tro len
            if idx != 0:
                k = 0
                while len(right_hand_side[0]) < len(freq_set) - 1:
                    ck = generate_candidate_k(right_hand_side, k=k)
                    rules, right_hand_side = compute_conf(freq_items, item_support_dict,
                                                          freq_set, ck, min_confidence)
                    association_rules.extend(rules)
                    k += 1

    return association_rules


if __name__ == "__main__":
    X, items = parse("data/a1_market_sample.csv")
    X = np.array(X)
    freq_items, item_support_dict = apriori(X, min_support=0.01)
    association_rules = create_rules(freq_items, item_support_dict, min_confidence=0.5)
    print(association_rules[:5])
