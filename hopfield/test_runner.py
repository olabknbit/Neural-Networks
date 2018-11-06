from util import read_file
import hopfield_net
from tabulate import tabulate


def main():
    prefix = 'hopfield/projekt2/'
    file_names = [
        ('animals-14x9.csv', (14, 9)), ('large-25x25.csv', (25, 25)), ('large-25x25.plus.csv', (25, 25)),
                  ('letters-8x12.csv', (8, 12)), ('letters-14x20.csv', (14, 20)), ('letters-abc-8x12.csv', (8, 12)),
                  ('small-7x7.csv', (7, 7))]

    percs = [0.05, 0.1, 0.15, 0.2]
    actions = ['sync']
    tabs = []
    for file_det in file_names:
        file_name, (width, height) = file_det

        images = read_file(prefix + file_name)

        for action in actions:
            name = file_name + '-' + action
            print(name)
            tab = [name]
            for perc in percs:
                flip = (int)(width * height * perc)
                accuracy = hopfield_net.run(images, width, height, 1, flip, False, 0, width * height * 20, action == 'sync')
                tab.append(accuracy)
            tabs.append(tab)

    headers = ['Dataset'] + [(str(p * 100) + '%') for p in percs]
    print(tabulate(tabs, headers=headers))


if __name__ == "__main__":
    main()
