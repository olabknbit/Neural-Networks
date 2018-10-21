from util import read_network_layers_from_file, write_network_to_file


def main2():
    import argparse

    parser = argparse.ArgumentParser(description='Neural Network framework.')
    parser.add_argument('--file', type= str, help='Name of a file containing graph')

    args = parser.parse_args()
    layers, output_classes = read_network_layers_from_file(args.file)

    import visualize
    visualize.main(layers, output_classes)

if __name__ == "__main__":
    main2()
