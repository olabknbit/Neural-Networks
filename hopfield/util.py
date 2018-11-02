import matplotlib.pyplot as plt
import numpy as np


# def generate_plot(table, width, height, ax, title=''):
#     from matplotlib.colors import ListedColormap
#
#     image = np.empty((width, height))
#     iter = 0
#     for i in range(width):
#         for j in range(height):
#             image[i][j] = table[iter]
#             iter += 1
#
#     cmap = ListedColormap(['k', 'w'])
#     fig3, ax3 = plt.subplots()
#     ax3.matshow(image, cmap=cmap)
#
#
#
#     cid = fig3.canvas.mpl_connect('button_press_event', lambda event: onclick(event, table, width, height))
#     plt.title(title)
#
#
# def onclick(event, table, width, height):
#     print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
#           ('double' if event.dblclick else 'single', event.button,
#            event.x, event.y, event.xdata, event.ydata))
#     table[int(0.5+event.xdata)*height + int(0.5+event.ydata)] *=-1
#
# # Printing table with {-1 white,1 black} value
# # table is list, image is 2-dim table
# # height and width are from args
# def print_image(table, width, height, title='', multiple=False, show=True):
#     if multiple:
#         fig = plt.figure(figsize=(8, 8))
#         columns = 5
#         rows = 5
#         for i, img in enumerate(table):
#             img, title = img
#             ax = fig.add_subplot(rows, columns, i + 1)
#             generate_plot(img, width, height, ax, title=title)
#     else:
#
#         generate_plot(table, width, height, plt, title=title)
#     if show:
#
#         plt.show()


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def print_image(self, image, title):
        a=1
        # print_image(image, self.width, self.height, title)

    def print_images(self, images):
        a=2
        # print_image(images, self.width, self.height, multiple=True)


def read_file(filename):
    with open(filename, 'r') as file:
        rows = file.readlines()
        images = [eval(row) for row in rows[0:]]
        return images



class Printer:
    def __init__(self, width, height, table2):
        self.width = width
        self.height = height
        self.table2 = table2
    def generate_plot2(self,  ax, title=''):
        from matplotlib.colors import ListedColormap

        image = np.empty((self.height, self.width))
        iter = 0
        for i in range(self.height):
            for j in range(self.width):
                image[i][j] = self.table2[iter]
                iter += 1

        cmap = ListedColormap(['k', 'w'])
        fig3, ax3 = plt.subplots()
        ax3.matshow(image, cmap=cmap)



        cid = fig3.canvas.mpl_connect('button_press_event', lambda event: self.onclick2(event))
        plt.title(title)


    def onclick2(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        self.table2[int(0.5+event.xdata) + int(0.5+event.ydata) *self.width] *=-1
        plt.close()
        self.print_image()

    # Printing table with {-1 white,1 black} value
    # table is list, image is 2-dim table
    # height and width are from args
    def print_image(self, title='', multiple=False, show=True):

        self.generate_plot2(plt, title=title)
        if show:

            plt.show()
