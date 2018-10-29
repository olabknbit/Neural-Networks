from numpy import random

def main():
    import argparse

    # parser = argparse.ArgumentParser(description='Neural Network framework.')
    # parser.add_argument('action', choices=['async', 'sync'],
    #                     help='Choose mode either \'async\' or \'sync\'.')
    #
    # parser.add_argument('--train_filename', type=str, help='Name of a file containing training data', required=False)
    # parser.add_argument('--test_filename', type= str, help='Name of a file containing testing data')
    # parser.add_argument('--create_nn', nargs='*', type=int,
    #                     help='When creating a nn from scratch; number of neurons for each layer',
    #                     required=False)
    #
    # parser.add_argument('--seed', type=int, help='Random seed int', required=False, default=1)
    #
    #
    # args = parser.parse_args()

    # Seed the random number generator
    # random.seed(args.seed)

    from util import printTable
    tab = ([1, -1, -1, 1], [1, 1,1,1], [1, -1,1, -1], [1, -1,1, -1])
    printTable(tab)



if __name__ == "__main__":
    main()
