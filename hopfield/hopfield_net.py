import numpy as np
from util import ImagePrinter


def _get_weigths(x_input):
    x_input = np.array(x_input)
    x_input_t = x_input.transpose()
    I = np.identity(len(x_input))
    return np.outer(x_input, x_input_t) - I


class Net:
    def __init__(self, image_printer):
        self.image_printer = image_printer

    def _initialize_params(self, x_inputs, bias):
        self.weights = _get_weigths(x_inputs[0])
        self.bias = bias
        for x_input in x_inputs[1:]:
            self.weights += _get_weigths(x_input)
        self.weights = self.weights / len(x_inputs)

    def recover_synchronous(self, x_input, visualize, plots, steps=10):
        x_input = np.array(x_input)

        last_energy = self.energy(x_input)
        for step, _ in enumerate(range(steps)):
            x_input = np.sign(np.dot(x_input, self.weights) - self.bias) # what if np.dot product is 0? what does sign do?
            e = self.energy(x_input)

            if e.__eq__(last_energy):
                break
            if visualize:
                print(x_input)
                print(e)
                plots.append((x_input, ('step ' + str(step))))
            last_energy = e

        return x_input

    def energy(self, x_input):
        return - x_input.dot(self.weights).dot(x_input) + np.sum(x_input * self.bias)


def flip_bits(image, bits, width, length, seed):
    import random
    random.seed(seed)

    image = np.array(image)
    for _ in range(bits):
        i = random.randint(0, width * length - 1)
        image[i] = -1 * image[i]
    return image


def run(images, width, length, seed, flip, visualize, bias, steps):
    ip = ImagePrinter(width, length)
    model = Net(ip)
    model._initialize_params(images, bias)

    for image in images:
        fuzzy_image = flip_bits(image, flip, width, length, seed)
        plots = []
        if visualize:
            plots.append((image, 'original'))
            plots.append((fuzzy_image, 'fuzzy'))
        e = model.energy(fuzzy_image)
        print(e)
        t = model.recover_synchronous(fuzzy_image, visualize, plots, steps=steps)
        if visualize:
            plots.append((tuple(t), 'output'))
        e = model.energy(t)
        ip.print_images(plots)
        print(e)
