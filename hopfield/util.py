# Printing table with {-1 white,1 black} value
# table is list, image is 2-dim table
# length and width are from args
def print_image(table, length, width):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    image = np.empty((length, width))
    iter=0
    for i in range(length):
        for j in range(width):
            image[i][j] = table[iter]
            iter += 1

    cmap = ListedColormap(['k', 'w'])
    plt.matshow(image, cmap=cmap)
    plt.show()


def read_file(filename):
    with open(filename, 'r') as file:
        rows = file.readlines()
        images = [eval(row) for row in rows[0:]]
        return images
