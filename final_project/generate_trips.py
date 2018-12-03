import random
PREFIX = 'final_project/data/'
DAY_LENGTH = 500


def get_routes_filename(lines, stops):
    return PREFIX + 'routes-' + str(lines) + '-lines-' + str(stops) + '-stops.txt'


def get_trips_filename(lines, stops, trips, transfers):
    return PREFIX + 'trips-' + str(lines) + '-lines-' + str(stops) + '-stops-' + str(trips) + '-M-' + str(transfers) + 'transfers.txt'


# Get starting params.
def get_random_stop_and_line_and_start_time(stops, lines):
    line = lines[random.randint(0, len(lines) - 1)]
    r = random.randint(0, len(stops[line]) - 1)
    stop = stops[line].keys()[r]
    start_time = random.randint(0, 500)
    return stop, line, start_time


# Go to another stop on THE SAME line.
def go_somewhere_from_stop_using_line(stops, stop, line):
    def should_continue():
        return random.random() < 0.5

    _, stop = stops[line][stop]
    while should_continue() and stop in stops[line].keys():
        _, stop = stops[line][stop]
    return stop


def get_transfer_lines(stops, lines, stop, line):
    transfer_lines = []
    for l in lines:
        if stop in stops[l].keys() and l != line:
            transfer_lines.append(l)
    return transfer_lines


class Trip:
    def __init__(self, start_time, stop):
        self.start_time = start_time
        self.stops = [stop]
        self.lines = []

    def append(self, stop, line):
        self.stops.append(stop)
        self.lines.append(line)

    def write_to_file(self, file):
        file.write(str(self.start_time) + ',' + self.stops[0])
        for line, stop in zip(self.lines, self.stops[1:]):
            file.write(',' + line + ',' + stop)
        file.write('\n')


def generate_trips(routes_filename, trips_filename, trips_m, transfers_max):
    import util
    _, stops, lines, first_stops = util.get_routes_parsed_info(routes_filename)

    with open(trips_filename, 'w') as trips_file:
        for i in range(trips_m):
            stop, line, start_time = get_random_stop_and_line_and_start_time(stops, lines)
            trip = Trip(start_time, stop)
            n_transfers = 0
            while n_transfers < transfers_max:
                stop = go_somewhere_from_stop_using_line(stops, stop, line)
                transfer_lines = get_transfer_lines(stops, lines, stop, line)
                trip.append(stop, line)
                # If transfer is not possible:
                if not transfer_lines:
                    break
                else:
                    line = transfer_lines[random.randint(0, len(transfer_lines) - 1)]
            trip.write_to_file(trips_file)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Neural Network framework.')
    parser.add_argument('--routes_lines', '-rl', type=int, default=3, help='Number of lines in routes file')
    parser.add_argument('--routes_stops', '-rs', type=int, default=6, help='Max number of stops in routes file')
    parser.add_argument('--trips_m', '-tm', type=int, default=10, help='How many trips')
    parser.add_argument('--trips_transfers', '-tt', type=int, default=2, help='Max number of transfers')
    parser.add_argument('--seed', type=int, help='Random seed int', required=False, default=1)

    args = parser.parse_args()

    # Seed the random number generator
    random.seed(args.seed)

    routes_filename = get_routes_filename(args.routes_lines, args.routes_stops)
    trips_filename = get_trips_filename(args.routes_lines, args.routes_stops, args.trips_m, args.trips_transfers)
    generate_trips(routes_filename, trips_filename, args.trips_m, args.trips_transfers)


if __name__ == "__main__":
    main()
