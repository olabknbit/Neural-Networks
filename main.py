from numpy import random


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Neural Network framework.')
    parser.add_argument('action', choices=['regression', 'classification'],
                        help='Choose mode either \'regression\' or \'classification\'.')

    parser.add_argument('--train_filename', type=str, help='Name of a file containing training data', required=False)
    parser.add_argument('--test_filename', type= str, help='Name of a file containing testing data')
    parser.add_argument('--create_nn', nargs='+', type=int,
                        help='When creating a nn from scratch; number of neurons for each layer',
                        required=False)
    parser.add_argument('--save_nn',  type=str, help='Name of a file to save trained model to.')
    parser.add_argument('-e', '--number_of_epochs', type=int, help='Number of epochs (iterations) for the NN to run',
                        required=False, default=10000)
    parser.add_argument('--read_nn', type=str, help='When reading existing nn from a file; filename')
    parser.add_argument('-v', '--visualize_every', type=int,
                        help='How ofter (every n iterations) print neuron\'s weights.',
                        required=False, default=1000)
    parser.add_argument('--l_rate', type=float, help='Learning rate', required=False, default=0.001)

    parser.add_argument('--seed', type=int, help='Random seed int', required=False, default=1)

    args = parser.parse_args()

    # Seed the random number generator
    random.seed(args.seed)

    if args.create_nn is None and args.read_nn is None:
        print('Either \'--create_nn\' or \'--read_nn\' has to be provided.')
        exit(1)

    if args.train_filename is None and args.save_nn is not None:
        print('\'--save_nn\' cannot be provided when \'--train_filename\' is not provided.')
        exit(1)

    if args.train_filename is None and args.create_nn is not None:
        print('\'--create_nn\' cannot be provided when \'--train_filename\' is not provided.')
        exit(1)

    if args.action == 'regression':
        import regression
        regression.main(args.train_filename, args.test_filename, args.create_nn, args.save_nn, args.read_nn,
                        args.number_of_epochs, args.visualize_every, args.l_rate)
    elif args.action == 'classification':
        import classification
        classification.main(args.train_filename, args.test_filename, args.create_nn, args.save_nn, args.read_nn,
                        args.number_of_epochs, args.visualize_every, args.l_rate)
    else:
        print('Sorry, first positional argument has to be either \'regression\' or \'classification\'.')
        exit(1)


if __name__ == "__main__":
    main()
