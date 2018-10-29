# import networkx as nx
# import matplotlib.pyplot as plt
# import re
# import sys
import os


def visualize(layers, output_classes, epoch):
    if os.name == 'nt':
        os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    from graphviz import Digraph
    f = Digraph(epoch)
    f.attr(rankdir='LR', size='50')

    i = 0
    prev = []
    next = []

    for o in layers[0][0]['weights']:
        f.node(str(i))
        next.append(str(i))
        i += 1

    prev = next
    next = []

    for row in layers:
        for k in row:
            f.node('Σ' + str(i))
            f.node(str(i))
            label = str('%.3f' % k['output'])
            f.edge('Σ' + str(i),  str(i), xlabel= str(label), fontsize='8', penwidth='0.01' )
            next.append(str(i))
            j=0
            for z in prev:
                label = str('%.3f' % k['weights'][j])
                f.edge(str(z), 'Σ' + str(i), xlabel= str(label), fontsize='8', penwidth='0.01' )
                j+=1
            i += 1
        prev = next
        next = []

    f.view()
def visualize2(layers, output_classes):
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()

    i = 0
    prev = []
    next = []

    for o in layers[0][0]['weights']:
        G.node(str(i))
        next.append(str(i))
        i += 1

    prev = next
    next = []


    for row in layers:
        for k in row:
            G.node(str(i))
            # G.node('Σ' + str(i))
            label = str('%.3f' % k['output'])
            # G.add_edge(str(i), 'Σ' + str(i),  weight = label )
            # next.append('Σ' + str(i))
            next.append(str(i))
            j=0
            for z in prev:
                label = str('%.3f' % k['weights'][j])
                G.add_edge(str(z), str(i), weight = label)
                j+=1
            i += 1
        prev = next
        next = []
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] != 0]
    pos = nx.circular_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=50)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge,
                           width=1)
    # nx.draw_networkx_edges(G, pos, edgelist=esmall,
    #                        width=6, alpha=0.5, edge_color='b', style='dashed')
    edge_labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)

    plt.axis('off')
    plt.show()
def main(layers, output_classes, epoch):
    visualize(layers, output_classes, epoch)
