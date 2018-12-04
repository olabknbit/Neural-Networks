PREFIX = 'final_project/data/'
DAY_LENGTH = 500


def get_routes_filename(lines, stops):
    return PREFIX + 'routes-' + str(lines) + '-lines-' + str(stops) + '-stops.txt'


def get_trips_filename(lines, stops, trips, transfers):
    return PREFIX + 'trips-' + str(lines) + '-lines-' + str(stops) + '-stops-' + str(trips) + '-M-' + str(transfers) + \
           '-transfers.txt'


def str_buses(buses):
    return '-'.join([str(bus) for bus in buses])


def get_hours_filename(n, buses):
    return PREFIX + 'hours-' + str(n) + '-' + str_buses(buses) + '.txt'


def get_mode_filename(lines, stops, trips, transfers, n, buses, mode='train', ext='txt'):
    return PREFIX + mode + '_data-' + str(lines) + '-lines-' + str(stops) + '-stops-' + str(trips) + '-M-' + \
           str(transfers) + '-transfers-' + str(n) + '-N-' + str_buses(buses) + '-buses.' + ext


def get_train_data_filename(lines, stops, trips, transfers, n, buses):
    return get_mode_filename(lines, stops, trips, transfers, n, buses, mode='train')


def get_test_data_filename(lines, stops, trips, transfers, n, buses):
    return get_mode_filename(lines, stops, trips, transfers, n, buses, mode='test')


def get_savefig_filename(lines, stops, trips, transfers, n, buses):
    return get_mode_filename(lines, stops, trips, transfers, n, buses, mode='savefig', ext='png')


def get_save_nn_filename(lines, stops, trips, transfers, n, buses):
    return get_mode_filename(lines, stops, trips, transfers, n, buses, mode='nn')


def get_routes_parsed_info(routes_filename):
    stops = dict()
    lines = []
    first_stops = dict()
    with open(routes_filename, 'r') as file:
        routes = file.readlines()

        for route in routes:
            str_splt = list(map(str.strip, route.split(',')))
            line = dict()
            line_id = str_splt[0]
            first_stop = str_splt[2]

            first_stops[line_id] = first_stop

            lines.append(line_id)

            i = 2
            while i < len(str_splt) - 2:
                stop_id = str_splt[i]
                t = int(str_splt[i + 1])
                next_stop_id = str_splt[i + 2]
                line[stop_id] = (t, next_stop_id)
                i += 2

            stops[line_id] = line
    return routes, stops, lines, first_stops
