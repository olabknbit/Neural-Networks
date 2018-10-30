import matplotlib.pyplot as plt
import numpy as np


def generate_plot(table, length, width, ax, title=''):
    from matplotlib.colors import ListedColormap

    image = np.empty((length, width))
    iter = 0
    for i in range(length):
        for j in range(width):
            image[i][j] = table[iter]
            iter += 1

    cmap = ListedColormap(['k', 'w'])
    ax.matshow(image, cmap=cmap)
    plt.title(title)


# Printing table with {-1 white,1 black} value
# table is list, image is 2-dim table
# length and width are from args
def print_image(table, length, width, title='', multiple=False, show=True):
    if multiple:
        fig = plt.figure(figsize=(8, 8))
        columns = 5
        rows = 5
        for i, img in enumerate(table):
            img, title = img
            ax = fig.add_subplot(rows, columns, i + 1)
            generate_plot(img, length, width, ax, title=title)
    else:
        generate_plot(table,width, length, plt, title=title)
    if show:
        plt.show()


class ImagePrinter:
    def __init__(self, width, length):
        self.width = width
        self.length = length

    def print_image(self, image, title):
        print_image(image, self.width, self.length, title)

    def print_images(self, images):
        print_image(images, self.width, self.length, multiple=True)


def read_file(filename):
    with open(filename, 'r') as file:
        rows = file.readlines()
        images = [eval(row) for row in rows[0:]]
        return images
