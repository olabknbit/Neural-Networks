PREFIX = 'final_project/data/'
GEN = 'gen/'
HELP = 'help/'
PLOTS = 'plots/'
TRAIN_TEST = 'train_test/'
NN = 'nn/'
DAY_LENGTH = 500


def get_routes_filename(lines, stops):
    return PREFIX + 'routes-' + str(lines) + '-lines-' + str(stops) + '-stops.txt'


def get_trips_filename(lines, stops, trips, transfers):
    return PREFIX + GEN + HELP + 'trips-' + str(lines) + '-lines-' + str(stops) + '-stops-' + str(trips) + '-M-' + \
           str(transfers) + '-transfers.txt'


def str_list(l):
    return '-'.join([str(e) for e in l])


def get_hours_filename(n, buses):
    return PREFIX + GEN + HELP + 'hours-' + str(n) + '-' + str_list(buses) + '.txt'


def get_mode_filename(lines, stops, trips, transfers, n, buses, mode='train', nn=None, ext='txt'):
    if nn is not None:
        nn = '-' + str_list(nn) + '-nn'
    else:
        nn = ''
    return PREFIX + GEN + mode + str(lines) + '-lines-' + str(stops) + '-stops-' + str(trips) + \
           '-M-' + str(transfers) + '-transfers-' + str(n) + '-N-' + str_list(buses) + '-buses' + nn + '.' + ext


def get_train_data_filename(lines, stops, trips, transfers, n, buses):
    return get_mode_filename(lines, stops, trips, transfers, n, buses, mode=TRAIN_TEST + 'train_data-')


def get_test_data_filename(lines, stops, trips, transfers, n, buses):
    return get_mode_filename(lines, stops, trips, transfers, n, buses, mode=TRAIN_TEST + 'test_data-')


def get_savefig_filename(lines, stops, trips, transfers, n, buses, nn):
    return get_mode_filename(lines, stops, trips, transfers, n, buses, mode=PLOTS + 'savefig', nn=nn, ext='png')


def get_save_nn_filename(lines, stops, trips, transfers, n, buses, nn):
    return get_mode_filename(lines, stops, trips, transfers, n, buses, mode=NN + 'nn', nn=nn)


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
