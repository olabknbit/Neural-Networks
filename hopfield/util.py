# Printing table with {-1 white,1 black} value
def printTable(table):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(['k', 'w'])
    plt.matshow(table, cmap=cmap)
    plt.show()
