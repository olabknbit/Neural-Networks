def main():
    import argparse

    parser = argparse.ArgumentParser(description='Neural Network framework.')
    parser.add_argument('--train_filename', type=str, help='Name of a file containing training data', required=True)
    parser.add_argument('-w','--width', type=int, help='Image width', required=True)
    parser.add_argument('-l','--length',  type=int, help='Image length', required=True)

    args = parser.parse_args()

    from util import print_image, read_file
    images = read_file(args.train_filename)

    # print(args.width)
    # print(args.length)
    # for image in images:
    #     print_image(image, args.width, args.length)

    import hopfield_net
    hopfield_net.run(images, args.width, args.length)

if __name__ == "__main__":
    main()
