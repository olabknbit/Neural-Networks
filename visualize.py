# import networkx as nx
# import matplotlib.pyplot as plt
# import re
# import sys
import os


def visualize(layers, output_classes):

    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    
    from graphviz import Digraph
    f = Digraph('neural_network')
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
            f.node(str(i))
            f.node('Σ' + str(i))
            label = str('%.3f' % k['output'])
            f.edge(str(i), 'Σ' + str(i), xlabel= str(label), fontsize='8', penwidth='0.01' )
            next.append('Σ' + str(i))
            j=0
            for z in prev:
                label = str('%.3f' % k['weights'][j])
                f.edge(str(z), str(i), xlabel= str(label), fontsize='8', penwidth='0.01' )
                j+=1
            i += 1
        prev = next
        next = []

    f.view()

def main(layers, output_classes):
    visualize(layers, output_classes)
