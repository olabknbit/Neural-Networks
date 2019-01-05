def get_times_at_stops(stops_orig, lines, first_stops, hours):
    # create a dict of a form: key = line_id; value another dict where key = stop_id, value = list of times when buses
    # of that line stop at this stop
    buses_at_stops = dict()
    for hour, line in zip(hours, lines):
        bas = dict()
        stop = first_stops[line]
        stops = stops_orig[line]

        t = 0
        while stop in stops.keys():
            ss = []
            for h in hour:
                ss.append(h + t)
            bas[stop] = ss
            t_new, s_new = stops[stop]
            stop = s_new
            t += t_new
        buses_at_stops[line] = bas

    return buses_at_stops


def get_next_bus_time(times, start_time):
    for time in times:
        if time >= start_time:
            return time
    return None


def get_time_at_next_stop(buses_at_stops, stops_orig, line_id, stop, start_time, final_stop):
    times = buses_at_stops[line_id][stop]
    nbt = get_next_bus_time(times, start_time)
    if nbt is None:
        return None

    start_time = nbt
    ttg, ns = stops_orig[line_id][stop]
    start_time += ttg
    if ns == final_stop:
        return start_time

    return get_time_at_next_stop(buses_at_stops, stops_orig, line_id, ns, start_time, final_stop)


def run(stops_orig, lines, first_stops, hours_l, trips):
    results = []
    for hours in hours_l:
        buses_at_stops = get_times_at_stops(stops_orig, lines, first_stops, hours)
        overall_t = 0
        for start_time, stop, transfers in trips:
            t = start_time

            while stop in transfers.keys():
                line_id, next_stop = transfers[stop]
                start_time = get_time_at_next_stop(buses_at_stops, stops_orig, line_id, stop, start_time, next_stop)
                if start_time is None:
                    # penalty bc someone just got stranded at the bus stop.
                    start_time = 100000
                    break
                stop = next_stop

            overall_t += start_time - t
        results.append(overall_t / len(trips))
    return results


def run_simulator(routes_filename, trips_filename, hours_filename):
    import util
    routes, stops, lines, first_stops = util.get_routes_parsed_info(routes_filename)

    with open(hours_filename, 'r') as file:
        hours_l = file.readlines()
        hours_l = map(eval, hours_l)

    with open(trips_filename, 'r') as file:
        trips = file.readlines()

        passenger_trips = []
        for trip in trips:
            str_splt = list(map(str.strip, trip.split(',')))

            start_time = int(str_splt[0])
            first_stop = str_splt[1]

            transfers = dict()

            i = 1
            while i < len(str_splt) - 2:
                stop_id = str_splt[i]
                line_id = str_splt[i + 1]
                next_stop_id = str_splt[i + 2]
                transfers[stop_id] = (line_id, next_stop_id)
                i += 2

            passenger_trips.append((start_time, first_stop, transfers))

    if routes is None or hours_l is None or trips is None:
        print('Failed to read buses info or hours info or trips info', routes, hours_l, trips)
        exit(1)

    return run(stops, lines, first_stops, hours_l, passenger_trips)


def run_simulator_with_args(routes_lines, routes_stops, trips_m, trips_transfers, hours_n, hours_buses):
    from final_project.data.gen import util
    routes_filename = util.get_routes_filename(routes_lines, routes_stops)
    trips_filename = util.get_trips_filename(routes_lines, routes_stops, trips_m, trips_transfers)
    hours_filename = util.get_hours_filename(hours_n, hours_buses)

    run_simulator(routes_filename, trips_filename, hours_filename)


def main():
    import argparse, random

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

    # Seed the random number generator
    #random.seed(args.seed)

    run_simulator_with_args(args.routes_lines, args.routes_stops, args.trips_m, args.trips_transfers, args.hours_n,
                            args.hours_buses)


if __name__ == "__main__":
    main()
