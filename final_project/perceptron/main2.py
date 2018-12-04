from perceptron.util import read_network_layers_from_file


def main2():
    import argparse

    parser = argparse.ArgumentParser(description='Neural Network framework.')
    parser.add_argument('--file', type=str, help='Name of a file containing graph')

    args = parser.parse_args()
    layers, _ = read_network_layers_from_file(args.file)

    from perceptron import visualize
    visualize.main(layers, args.file)


if __name__ == "__main__":
    main2()
