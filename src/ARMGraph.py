import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.preprocessing.transactionencoder import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def arm(transactions, support, lift, head):
    te = TransactionEncoder()
    transactions_df = te.fit_transform(transactions)
    transactions_df = pd.DataFrame(transactions_df, columns=te.columns_)

    frequent_itemsets = apriori(transactions_df, min_support=0.05, use_colnames=True)

    ar = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.8)
    ar = ar.query("support >= {0} and lift >= {1}".format(support,
                                                          lift)).sort_values(by='lift',
                                                                             ascending=False).head(head)
    return ar


def drawGraph(ar, Multiplier=1000):
    G = nx.DiGraph()
    size_dict = {}
    color_dict = {}
    label_dict = {}
    for i in ar.index:
        ser = ar.loc[i]
        G.add_node(i)
        # node_list[i] = i
        size_dict[i] = ser['support'] * Multiplier
        color_dict[i] = ser['lift']
        label_dict[i] = ''
        for ant in list(ser['antecedents']):
            G.add_node(ant)
            G.add_edge(ant, i)
            size_dict[ant] = 0
            color_dict[ant] = 0
            label_dict[ant] = ant

        for j in list(ser['consequents']):
            G.add_node(j)
            size_dict[ j ] = 0
            color_dict[ j ] = 0
            label_dict[ j ] = j
            G.add_edge(i, j)

    node_list, size_list = zip(*size_dict.items())
    node_list, color_list = zip(*color_dict.items())

    pos = nx.kamada_kawai_layout(G)
    cmap = plt.cm.get_cmap('Reds')
    nx.draw_networkx_nodes(G, pos=pos,
                           nodelist=node_list,
                           node_size=size_list,
                           node_color=color_list,
                           alpha=0.5, with_labels=True, cmap=cmap)
    nx.draw_networkx_edges(G,pos=pos, edge_color='grey', alpha=0.5)
    nx.draw_networkx_labels(G, pos=pos,labels=label_dict, font_size=10)
    plt.show()


if __name__ == '__main__':
    transactions = [ [ 'milk', 'bread' ],
                     [ 'bread', 'nappy', 'beer', 'potato' ],
                     [ 'milk', 'nappy', 'beer', 'coke' ],
                     [ 'bread', 'milk', 'nappy', 'beer' ],
                     [ 'bread', 'milk', 'nappy', 'coke' ] ]

    ar = arm(transactions, 0.3, 1.2, 5)
    drawGraph(ar, Multiplier=1000)