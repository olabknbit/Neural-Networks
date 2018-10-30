import numpy as np
from util import ImagePrinter


def _get_weigths(x_input):
    x_input = np.array(x_input)
    x_input_t = x_input.transpose()
    return np.outer(x_input, x_input_t)


class Net:
    def __init__(self, image_printer):
        self.image_printer = image_printer

    def _initialize_params(self, x_inputs, bias):
        self.weights = _get_weigths(x_inputs[0])
        self.bias = bias
        for x_input in x_inputs[1:]:
            self.weights += _get_weigths(x_input)

        I = np.identity(len(x_inputs[0]))
        self.weights = self.weights - I
        self.weights = self.weights / len(x_inputs)

    def recover_synchronous(self, x_input, visualize, steps=10):
        x_input = np.array(x_input)

        last_energy = self.energy(x_input)
        for step, _ in enumerate(range(steps)):
            x_input = np.sign(np.dot(x_input, self.weights) - self.bias)
            e = self.energy(x_input)

            if e.__eq__(last_energy):
                break
            if visualize:
                print(x_input)
                print(e)
                self.image_printer.print_image(x_input, title=('step ' + str(step)))
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


def run(images, width, length, seed, flip, visualize, bias):
    ip = ImagePrinter(width, length)
    model = Net(ip)
    model._initialize_params(images, bias)

    for image in images[-3:]:
        fuzzy_image = flip_bits(image, flip, width, length, seed)
        if visualize:
            ip.print_image(image, title='original')
            ip.print_image(fuzzy_image, title='fuzzy')
        e = model.energy(fuzzy_image)
        print(e)
        t = model.recover_synchronous(fuzzy_image, visualize)
        if visualize:
            ip.print_image(tuple(t), 'output')
            ip.print_image(tuple(image), title='original')
        e = model.energy(t)
        print(e)
