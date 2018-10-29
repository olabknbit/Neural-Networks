import numpy as np
from util import print_image


def _get_weigths(x_input):
    x_input = np.array(x_input)
    x_input_t = x_input.transpose()
    I = np.identity(len(x_input))
    return np.outer(x_input, x_input_t) - I


class Net:
    def _initialize_params(self, x_inputs):
        self.weights = _get_weigths(x_inputs[0])
        for x_input in x_inputs[1:]:
            self.weights += _get_weigths(x_input)

    def recover_synchronous(self, x_input, steps=5):
        x_input = np.array(x_input)
        patterns = self.weights
        for _ in range(steps):
            # dot -> matmul instead?
            patterns = np.sign(np.dot(patterns, x_input))
        return patterns


def flip_bits(image, bits, width, length):
    import random

    image = np.array(image)
    for _ in range(bits):
        i = random.randint(0, width * length)
        image[i] = -1 * image[i]
    return image


def run(images, width, length):
    model = Net()
    model._initialize_params(images)

    for image in images:
        fuzzy_image = flip_bits(image, 5, width, length)
        print_image(fuzzy_image, width, length)
        i1 = tuple(model.recover_synchronous(image))
        print_image(i1, width, length)


