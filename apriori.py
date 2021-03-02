import numpy as np
import pandas as pd
import pickle
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
    @rtype: List[Set(frozen)]
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
    Generate list of  of size-(k+2) candidates from the  frequent item list (freq_item)

    @param freq_item: List of qualified item sets (item sets with support >= min_support
    @type freq_item:  List[]
    @param k: k common items in two (k+1) itemset
    @type k: int
    @return: List of frequece itemset, size = k + 2
    @rtype: List[frozen]
    """
    size_k_candidates = []

    # to generate candidates of size two (2-item sets)

    if k == 0:
        # if k = 0, size of itemset = 1 => size of common items = 0
        for f1, f2 in combinations(freq_item, 2):
            item = f1 | f2  # union of two sets
            size_k_candidates.append(item)

    # k > 0
    else:
        for f1, f2 in combinations(freq_item, 2):
            # if the two (k+1)-item sets both has k common items
            # then they will be combined to be the (k+2)-item candidate
            # k >= 1, gen list common items in two itemset size (k+1)
            intersection = f1 & f2
            # if size of common items = k
            if len(intersection) == k:
                item = f1 | f2 # union of two sets, expected size-itemset = k + 2
                if item not in size_k_candidates:
                    size_k_candidates.append(item)

    return size_k_candidates


def create_freq_item(transaction_list, size_k_candidates, min_support):
    """
    find candidates whose support >= min_support
    @param transaction_list: list of transactions with transaction items being coded by their sequential indexes
    @type transaction_list: List[Set[int]]
    @param size_k_candidates: size-k itemset
    @type size_k_candidates: List
    @param min_support: The minimum fraction of transactions an itemets needs to occur in to be deemed frequent
    @type min_support: float
    @return:
        freq_item: frequence itemset have support >= min_support
        item_support: itemset with support index               
    @rtype:
        freq_item: List[frozen]
        item_support: Dict, item: support, support is float, between 0 and 1
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
    n_row = len(transaction_list)

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
    pass in the transaction data and the minimum support threshold to obtain the frequent itemset. 
        Also store the support for each itemset, they will be used in the rule generation step
    @param transaction_list: list of transactions with transaction items being coded by their sequential indexes
    @type transaction_list: List[Set(int)]
    @param min_support: Minumum support of the itemsets returned. 
        The support is frequency of which the items in the rule appear together in the data set.
    @type min_support: Float between 0 and 1
    @return:
        freq_items: List of list k-itemset, k = 0, ..n
        item_support_dict: Dict of itemset, size k = 0, ..n
    @rtype:
        freq_items: List[List]
        item_support_dict: Dict, item: support, support is Float between 0 and 1
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
    Create the rules and returns the rules info and the rules's
        right hand side (used for generating the next round of rules) 
        if it surpasses the minimum confidence threshold
    @params:
        freq_items: List of list k-itemset, k = 0, ..n
        item_support_dict: Dict of itemset, size k = 0, ..n
        freq_set: itemset
        subsets: subset of freq_set
        min_confidence: The minimum confidence of the rules returned. Given a rule X -> Y, the
            confidence is the probability of Y, given X, i.e. P(Y|X) = conf(X -> Y)
    @type:
        freq_items: List[List]
        item_support_dict: Dict, item: support, support is Float between 0 and 1
        freq_set: List[frozenset]
        subsets: frozenset
        min_confidence: Float between 0 and 1
    @return:
        rules: List of rule, rule (Typle) include:
            - lhs: left hand side
            - rhs: right hand size
            - conf: confidence score
            - lift: lift score
        right_hand_size: List of rhs
    @rtype:
        rules: List[Tuple]
        right_hand_size: List[frozenset]
    """
    # rule X->Y-X
    # lhs = X
    # rhs = Y-X
    rules = []
    right_hand_side = []

    # create the left hand side of the rule
    # and add the rules if it's greater than
    # the confidence threshold
    for rhs in subsets:
        # create the left hand side of the rule
        lhs = freq_set - rhs
        # compute confidence of rule
        conf = item_support_dict[freq_set] / item_support_dict[lhs]
        # if conf >= min_confidence, add rule to List of rule
        if conf >= min_confidence:
            lift = conf / item_support_dict[rhs]
            # defince rule and add into List of rule returned
            rules_info = lhs, rhs, conf, lift
            rules.append(rules_info)
            right_hand_side.append(rhs)

    return rules, right_hand_side


def create_rules(freq_items, item_support_dict, min_confidence):
    """
    Create the association rules,
    rule contain rule, left hand side, right hand side, confidence, lift
    @param:
        freq_items: List of list k-itemset, k = 0, ..n
        item_support_dict: Dict of itemset, size k = 0, ..n
        min_confidene: Minimum of confidence score
    @type:
        freq_items: List[List]
        item_support_dict: Dict, item: support, support is Float between 0 and 1
        min_confidence: Float between 0 to 1
    @return: List of rules
    @rtype: List[tuple]
    """
    association_rules = []

    # for the list that stores the frequent items, loop through
    # the second element to the one before the last to generate the rules
    # because the last one will be an empty list. It's the stopping criteria
    # for the frequent itemset generating process and the first one are all
    # single element frequent itemset, which can't perform the set
    # operation X -> Y - X
    for idx, freq_item in enumerate(freq_items[1:(len(freq_items) - 1)]):
        for freq_set in freq_item:
            # start with creating rules for single item on
            # the right hand side
            subsets = [frozenset([item]) for item in freq_set]
            rules, right_hand_side = compute_conf(
                freq_items, item_support_dict, freq_set, subsets, min_confidence)
            association_rules.extend(rules)

            # starting from 3-itemset, loop through each length item
            # to create the rules, as for the while loop condition,
            # e.g. suppose you start with a 3-itemset {2, 3, 5} then the 
            # while loop condition will stop when the right hand side's
            # item is of length 2, e.g. [ {2, 3}, {3, 5} ], since this
            # will be merged into 3 itemset, making the left hand side
            # null when computing the confidence
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
    X, items = parse("data/a1_market_basket_optimisation.csv")
    X = np.array(X)
    freq_items, item_support_dict = apriori(X, min_support=0.01)
    association_rules = create_rules(freq_items, item_support_dict, min_confidence=0.05)
    print(association_rules[-5:])
