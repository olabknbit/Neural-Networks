def generate_test_train_data(routes_filename, trips_filename, hours_filename, train_data_filename, test_data_filename):
    import simulator
    results = simulator.run_simulator(routes_filename, trips_filename, hours_filename)

    def write_data(data, file):
        for result, line in data:
            hours_set = eval(line)
            [file.write(str(hour) + ',') for hours in hours_set for hour in hours]
            file.write(str(result) + '\n')

    with open(hours_filename, 'r') as hours_file, open(train_data_filename, 'w') as train_data_file, open(
            test_data_filename, 'w') as test_data_file:
        train_n = int(len(results) * 0.4)
        data = list(zip(results, hours_file.readlines()))
        write_data(data[:train_n], train_data_file)
        write_data(data[train_n:], test_data_file)


def generate_test_train_data_without_filenames(routes_lines, routes_stops, trips_m, trips_transfers, hours_n,
                                               hours_buses):
    import util
    routes_filename = util.get_routes_filename(routes_lines, routes_stops)
    trips_filename = util.get_trips_filename(routes_lines, routes_stops, trips_m, trips_transfers)
    hours_filename = util.get_hours_filename(hours_n, hours_buses)
    train_data_filename = util.get_train_data_filename(routes_lines, routes_stops, trips_m, trips_transfers, hours_n,
                                                       hours_buses)
    test_data_filename = util.get_test_data_filename(routes_lines, routes_stops, trips_m, trips_transfers, hours_n,
                                                     hours_buses)

    generate_test_train_data(routes_filename, trips_filename, hours_filename, train_data_filename, test_data_filename)


def main():
    import argparse, util
    parser = argparse.ArgumentParser(description='Neural Network framework.')
    parser.add_argument('--routes_lines', '-rl', type=int, default=3, help='Number of lines in routes file')
    parser.add_argument('--routes_stops', '-rs', type=int, default=6, help='Max number of stops in routes file')
    parser.add_argument('--trips_m', '-tm', type=int, default=10, help='How many trips')
    parser.add_argument('--trips_transfers', '-tt', type=int, default=2, help='Max number of transfers')
    parser.add_argument('--hours_buses', '-hb', nargs='*', type=int, default=[1, 1, 1],
                        help='How many buses of each line')
    parser.add_argument('--hours_n', '-hn', type=int, default=1000, help='How many different sets of hours')

    parser.add_argument('--seed', type=int, help='Random seed int', required=False, default=1)

    args = parser.parse_args()

    routes_filename = util.get_routes_filename(args.routes_lines, args.routes_stops)
    trips_filename = util.get_trips_filename(args.routes_lines, args.routes_stops, args.trips_m, args.trips_transfers)
    hours_filename = util.get_hours_filename(args.hours_n, args.hours_buses)
    train_data_filename = util.get_train_data_filename(args.routes_lines, args.routes_stops, args.trips_m,
                                                       args.trips_transfers, args.hours_n, args.hours_buses)
    test_data_filename = util.get_test_data_filename(args.routes_lines, args.routes_stops, args.trips_m,
                                                     args.trips_transfers, args.hours_n, args.hours_buses)

    generate_test_train_data(routes_filename, trips_filename, hours_filename, train_data_filename, test_data_filename)


if __name__ == "__main__":
    main()
