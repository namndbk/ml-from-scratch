import apriori


def get_random_10_rules(path):
    pass


def save_rules(X, items, path, min_support=0.01, min_confidence=0.05):
    """
    Save rules as file.txt
    """
    freq_items, item_support_dict = apriori.apriori(X, min_support=min_support)
    association_rules = apriori.create_rules(freq_items, item_support_dict, min_confidence=min_confidence)
    index_items = {items[item]: item for item in items}
    rules = []
    for rule in association_rules:
        lhs = rule[0]
        rhs = rule[1]
        confidence = rule[2]
        lift = rule[-1]
        r = ", ".join(index_items[item] for item in lhs) + " ----> " + ", ".join(index_items[item] for item in rhs) \
            + "\tmin support = %.f" % min_support + "\tconfidence score = %.3f" % confidence + "\tlift score = %.3f" % lift
        rules.append(r)
    try:
        with open(path, "w") as f:
            for r in rules:
                f.write(str(r) + "\n")
        print("\tRules saved in file {}".format(path))
    except:
        print("\tNot save rules !")
                

if __name__ == "__main__":
    X, items = apriori.parse("data/a1_market_basket_optimisation.csv")
    save_rules(X, items, path="apriori.txt")

