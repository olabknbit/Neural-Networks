import matplotlib.pyplot as plt
import numpy as np


def generate_plot(table, width, height, ax, title=''):
    from matplotlib.colors import ListedColormap

    image = np.empty((height, width))
    iter = 0
    for i in range(height):
        for j in range(width):
            image[i][j] = table[iter]
            iter += 1

    cmap = ListedColormap(['k', 'w'])
    ax.matshow(image, cmap=cmap)
    plt.title(title)



# Printing table with {-1 white,1 black} value
# table is list, image is 2-dim table
# height and width are from args
def print_image(table, width, height, title='', multiple=False, show=True):
    if multiple:
        fig = plt.figure(figsize=(8, 8))
        columns = 5
        rows = 5
        for i, img in enumerate(table):
            img, title = img
            ax = fig.add_subplot(rows, columns, i + 1)
            generate_plot(img, width, height, ax, title=title)
    else:
        generate_plot(table, width, height, plt, title=title)
    if show:
        
        plt.show()


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def print_image(self, image, title):
        print_image(image, self.width, self.height, title)

    def print_images(self, images):
        print_image(images, self.width, self.height, multiple=True)


def read_file(filename):
    with open(filename, 'r') as file:
        rows = file.readlines()
        images = [eval(row) for row in rows[0:]]
        return images



class ResponsiveImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def generate_plot(self,  ax, title=''):
        from matplotlib.colors import ListedColormap

        image = np.empty((self.height, self.width))
        iter = 0
        for i in range(self.height):
            for j in range(self.width):
                image[i][j] = self.table[iter]
                iter += 1

        cmap = ListedColormap(['k', 'w'])
        fig, ax = plt.subplots()
        ax.matshow(image, cmap=cmap)

        cid = fig.canvas.mpl_connect('button_press_event', lambda event: self.onclick(event))
        plt.title(title)


    def onclick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        self.table[int(0.5+event.xdata) + int(0.5+event.ydata) *self.width] *=-1
        plt.close()
        self.print_image(self.table)

    # Printing table with {-1 white,1 black} value
    # table is list, image is 2-dim table
    # height and width are from args
    def print_image(self,table, title=''):
        self.table = table
        self.generate_plot(plt, title=title)
        plt.show()
