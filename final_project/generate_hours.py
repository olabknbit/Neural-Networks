import random, util


def generate_hours(buses, n, filename):
    with open(filename, 'w') as file:
        for _ in range(n):
            hours = []
            for bus in buses:
                line_hours = []
                for _ in range(bus):
                    line_hours.append(random.randint(0, util.DAY_LENGTH))
                hours.append(line_hours)
            file.write(str(hours))
            file.write('\n')


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Neural Network framework.')
    parser.add_argument('buses', nargs='*', type=int, default=[1, 1, 1], help='How many buses of each line')
    parser.add_argument('-n', type=int, default=1000, help='How many different sets of hours')
    parser.add_argument('--seed', type=int, help='Random seed int', required=False, default=1)

    args = parser.parse_args()

    # Seed the random number generator
    random.seed(args.seed)

    filename = util.get_hours_filename(args.n, args.buses)
    generate_hours(args.buses, args.n, filename)


if __name__ == "__main__":
    main()
