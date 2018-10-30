import numpy as np
from util import ImagePrinter


def _get_weigths(x_input):
    x_input = np.array(x_input)
    x_input_t = x_input.transpose()
    return np.outer(x_input, x_input_t)


class Net:
    def __init__(self, image_printer):
        self.image_printer = image_printer

    def _initialize_params(self, x_inputs):
        self.weights = _get_weigths(x_inputs[0])
        for x_input in x_inputs[1:]:
            self.weights += _get_weigths(x_input)

        I = np.identity(len(x_inputs[0]))
        self.weights = self.weights = I

    def recover_synchronous(self, x_input, visualize, steps=5):
        x_input = np.array(x_input)

        last_patterns = x_input
        for _ in range(steps):
            x_input = np.sign(np.dot(x_input, self.weights))

            if visualize:
                print(x_input)
                self.image_printer.print_image(x_input)
            if np.equal(x_input, last_patterns).all():
                break
            last_patterns = x_input

        return x_input


def flip_bits(image, bits, width, length, seed):
    import random
    random.seed(seed)

    image = np.array(image)
    for _ in range(bits):
        i = random.randint(0, width * length)
        image[i] = -1 * image[i]
    return image


def run(images, width, length, seed, flip, visualize):
    ip = ImagePrinter(width, length)
    model = Net(ip)
    model._initialize_params(images)

    for image in images[-1:]:
        fuzzy_image = flip_bits(image, flip, width, length, seed)
        if visualize:
            ip.print_image(fuzzy_image)
        i1 = tuple(model.recover_synchronous(image, visualize))
        # ip.print_image(i1)


