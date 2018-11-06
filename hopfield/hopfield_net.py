import numpy as np
from util import ImagePrinter, ResponsiveImagePrinter


def _get_weigths(x_input):
    x_input = np.array(x_input)
    I = np.identity(len(x_input))
    a = np.outer(x_input, x_input) - I
    return a


class Net:
    def __init__(self, image_printer):
        self.image_printer = image_printer

    def _initialize_params(self, x_inputs, bias):
        self.weights = _get_weigths(x_inputs[0])
        self.bias = bias
        self.train_more(x_inputs[1:], bias)

    def train_more(self, x_inputs, bias=0):
        for x_input in x_inputs:
            np.add(self.weights, _get_weigths(x_input), out=self.weights)
        if bias is not 0:
            self.weights = self.weights / len(x_inputs)

    def recover_synchronous(self, x_input, visualize, plots, width, height, steps=10):
        x_input = np.array(x_input)
        responsive_printer = None
        if visualize == -1:
            responsive_printer = ResponsiveImagePrinter(width, height)

        last_energy = self.energy(x_input)
        for step in range(steps):
            x_input = np.sign(np.dot(x_input, self.weights) - self.bias)
            e = self.energy(x_input)

            if e.__eq__(last_energy):
                break
            if visualize > 0 and step % visualize == 0:
                plots.append((x_input, ('step ' + str(step))))

            elif visualize == -1:
                responsive_printer.print_image(x_input, ('step ' + str(step)))
                x_input = responsive_printer.table

            last_energy = e
        if visualize == -1:
            responsive_printer.print_image(x_input, 'output')

        return x_input

    def recover_asynchronous(self, x_input, visualize, plots, width, height, steps=20):
        import random
        x_input = np.array(x_input)
        responsive_printer = ResponsiveImagePrinter(width, height)

        i = 0
        last_energy = self.energy(x_input)
        list_indexes = random.sample(range(0, len(x_input)), len(x_input))
        for step, _ in enumerate(range(steps)):
            temp_x_input = np.sign(np.dot(x_input, self.weights) - self.bias)
            x_input[list_indexes[step % len(x_input)]] = temp_x_input[list_indexes[step % len(x_input)]]

            if last_energy != self.energy(x_input):
                last_energy = self.energy(x_input)
                i = 0
            else:
                i += 1
            if i>=len(x_input):
                break

            if visualize == -1:
                responsive_printer.print_image(x_input, ('step ' + str(step)))
                x_input = responsive_printer.table

            elif visualize > 0 and step % visualize == 0:
                plots.append((np.copy(x_input), ('step ' + str(step))))

        if visualize == -1:
            responsive_printer.print_image(x_input, 'output')
        return x_input

    def energy(self, x_input):
        return - x_input.dot(self.weights).dot(x_input) + np.sum(x_input * self.bias)

    def test(self, images, width, height, seed, flip, visualize, steps, sync):
        correct = 0.
        for image in images:
            fuzzy_image = flip_bits(image, flip, width, height, seed)
            plots = []
            if visualize > 0:
                plots.append((image, 'original'))
                plots.append((fuzzy_image, 'fuzzy'))
            if sync:
                t = self.recover_synchronous(fuzzy_image, visualize, plots, width, height, steps=steps)
            else:
                t = self.recover_asynchronous(fuzzy_image, visualize, plots, width, height, steps=steps)

            if np.array_equal(t, image) or np.array_equal(t * 0.1, image):
                correct += 1

            if visualize > 0:
                plots.append((tuple(t), 'output'))
                self.ip.print_images(plots)

        accuracy = correct / len(images)
        return accuracy


def flip_bits(image, bits, width, height, seed):
    import random
    random.seed(seed)

    image = np.array(image)
    for _ in range(bits):
        i = random.randint(0, width * height - 1)

        image[i] = -1 * image[i]
    return image


def train(images, width, height, bias=0):
    ip = ImagePrinter(width, height)
    model = Net(ip)
    model._initialize_params(images, bias)
    return model


def run(images, width, height, seed, flip, visualize, steps, sync, bias=0):
    model = train(images, width, height, bias)
    return model.test(images, width, height, seed, flip, visualize, steps, sync)
