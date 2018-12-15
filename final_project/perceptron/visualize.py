import os


def visualize(layers, epoch):
    if os.name == 'nt':
        os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    from graphviz import Digraph
    f = Digraph(epoch)
    f.attr(rankdir='LR', size='50')

    i = 0
    next = []

    for _ in layers[0][0]['weights']:
        f.node(str(i))
        next.append(str(i))
        i += 1

    prev = next
    next = []

    for row in layers:
        for k in row:
            f.node('epsilon' + str(i))
            f.node(str(i))
            label = str('%.3f' % k['output'])
            f.edge('epsilon' + str(i), str(i), xlabel=str(label), fontsize='8', penwidth='0.01')
            next.append(str(i))
            j = 0
            for z in prev:
                label = str('%.3f' % k['weights'][j])
                f.edge(str(z), 'epsilon' + str(i), xlabel=str(label), fontsize='8', penwidth='0.01')
                j += 1
            i += 1
        prev = next
        next = []

    f.view()


def simple_vis(network):
    from graphviz import Digraph
    diagraph = Digraph()
    diagraph.attr(rankdir='LR', size='50')

    for neuron in network.neurons:
        diagraph.node(str(neuron.id))

    for n_out in network.neurons:
        for n_in, weight in n_out.in_ns.iteritems():
            label = str('%.3f' % weight)
            diagraph.edge(str(n_in.id), str(n_out.id), xlabel=str(label), fontsize='8', penwidth='0.01')

    diagraph.view()


def visualize2(layers):
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()

    i = 0
    next = []

    for _ in layers[0][0]['weights']:
        G.node(str(i))
        next.append(str(i))
        i += 1

    prev = next
    next = []

    for row in layers:
        for k in row:
            G.node(str(i))
            next.append(str(i))
            j = 0
            for z in prev:
                label = str('%.3f' % k['weights'][j])
                G.add_edge(str(z), str(i), weight=label)
                j += 1
            i += 1
        prev = next
        next = []
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] != 0]
    pos = nx.circular_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=50)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.axis('off')
    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Neural Network framework.')
    parser.add_argument('--file', type=str, help='Name of a file containing graph')

    args = parser.parse_args()
    from util import read_network_layers_from_file
    layers, _ = read_network_layers_from_file(args.file)

    visualize(layers, args.file)


if __name__ == "__main__":
    main()
