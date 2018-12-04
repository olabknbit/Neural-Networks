import matplotlib as mpl

mpl.use('TkAgg')
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
def print_image(table, width, height, title='', multiple=False, show=True, save=None):
    if multiple:
        fig = plt.figure(figsize=(8, 8))
        columns = 5
        rows = 5
        for i, img in enumerate(table):
            if len(img) == 2:
                img, title = img
            else:
                title = ''
            ax = fig.add_subplot(rows, columns, i + 1)
            generate_plot(img, width, height, ax, title=title)

    else:
        generate_plot(table, width, height, plt, title=title)
    if show:
        plt.show()
    if save != '':
        plt.savefig(save)


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def print_image(self, image, title, show=True, save=None):
        print_image(image, self.width, self.height, title, show=show, save=save)

    def print_images(self, images, show=True, save=None):
        print_image(images, self.width, self.height, multiple=True, show=show, save=save)


def read_file(filename):
    with open(filename, 'r') as file:
        rows = file.readlines()
        images = []
        for row in rows:
            if len(row) > 1:
                images.append(eval(row))
        return images


class ResponsiveImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def generate_plot(self, ax, title=''):
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
        self.table[int(0.5 + event.xdata) + int(0.5 + event.ydata) * self.width] *= -1
        plt.close()
        self.print_image(self.table)

    # Printing table with {-1 white,1 black} value
    # table is list, image is 2-dim table
    # height and width are from args
    def print_image(self, table, title=''):
        self.table = table
        self.generate_plot(plt, title=title)
        plt.show()


class BitmapConverter:
    def __init__(self, n):
        self.n = n
        self.how_many = int(self.n * self.n * 0.1)

    def convert(self, filename):
        from PIL import Image
        import numpy as np

        img = Image.open('hopfield/img/' + filename + '.jpg')
        ary = np.array(img)

        # Split the three channels
        r, g, b = np.split(ary, 3, axis=2)
        r = r.reshape(-1)
        g = r.reshape(-1)
        b = r.reshape(-1)

        # Standard RGB to grayscale
        bitmap = list(map(lambda x: 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2],
                          zip(r, g, b)))
        bitmap = np.array(bitmap).reshape([ary.shape[0], ary.shape[1]])
        bitmap = np.dot((bitmap > 128).astype(float), 255)

        im = Image.fromarray(bitmap.astype(np.uint8))
        im = im.resize((self.n, self.n))

        im.save('hopfield/bmp100/' + filename + '.bmp')

    def c(self, filename):
        from PIL import Image
        import numpy as np

        img = Image.open('hopfield/bmp100/' + filename + '.bmp')
        bitmap = np.array(img)
        return [-1 if int(pixel) is 0 else 1 for line in bitmap for pixel in line]

    def to_hopfield_input(self):
        with open('hopfield/projekt2/boats-100x100.csv', 'a') as file:
            for i in range(self.how_many):
                try:
                    file.write(','.join([str(pix) for pix in self.c(str(i))]))
                    file.write('\n')
                except:
                    None

    def convert_all(self):
        for i in range(self.how_many):
            try:
                self.convert(str(i))
            except:
                None

    def url_to_image(self, url):
        import urllib as urll
        import cv2
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        image = None
        try:
            resp = urll.urlopen(url)
            redirect = resp.geturl()
            if 'unavailable' not in redirect:
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        except:
            pass
        return image

    def get_images(self):
        from bs4 import BeautifulSoup
        import requests
        import cv2

        page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04194289")  # ship synset

        # BeautifulSoup is an HTML parsing library
        soup = BeautifulSoup(page.content,
                             'html.parser')  # puts the content of the website into the soup variable, each url on a different line
        str_soup = str(soup)  # convert soup to string so it can be split

        split_urls = str_soup.split('\r\n')  # split so each url is a different possition on a list

        img_rows, img_cols = self.n, self.n  # number of rows and columns to convert the images to
        input_shape = (
        img_rows, img_cols, 3)  # format to store the images (rows, columns,channels) called channels last

        n_of_training_images = self.how_many  # the number of training images to use
        for progress in range(min(n_of_training_images, len(split_urls))):  # store all the images on a directory
            # Print out progress whenever progress is a multiple of 20 so we can follow the
            # (relatively slow) progress
            if progress % 20 == 0:
                print(progress)
            if split_urls[progress] is not None:
                I = self.url_to_image(split_urls[progress])
                if I is not None and (len(I.shape)) == 3:  # check if the image has width, length and channels
                    filename = str(progress)
                    save_path = r'hopfield/img/' + filename + '.jpg'  # create a name of each image
                    cv2.imwrite(save_path, I)
