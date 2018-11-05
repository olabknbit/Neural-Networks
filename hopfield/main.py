def main():
    import argparse

    parser = argparse.ArgumentParser(description='Neural Network framework.')
    parser.add_argument('--action', help='Choose mode either \'sync\' or \'async\'.', required=False, default='sync')
    parser.add_argument('--train_filename', type=str, help='Name of a file containing training data', required=True)
    parser.add_argument('-w','--width', type=int, help='Image width', required=True)
    parser.add_argument('-he','--height',  type=int, help='Image height', required=True)
    parser.add_argument('--seed', type=int, help='Random seed int', required=False, default=1)
    parser.add_argument('--flip', type=int, help='How many bits to flip while testing', required=False, default=5)
    parser.add_argument('-vi','--visualize', type=int, help='Generate image after x run,if -1 each run responsive', required=False, default=1)
    parser.add_argument('--bias', type=int, help='Bias', required=False, default=1)
    parser.add_argument('--steps', type=int, help='Steps', required=False, default=10)

    args = parser.parse_args()

    from util import read_file
    images = read_file(args.train_filename)

    import hopfield_net
    hopfield_net.run(images, args.width, args.height, args.seed, args.flip, args.visualize, args.bias, args.steps, args.action == 'sync')


if __name__ == "__main__":
    main()
