def move_data(filename, d):
    m = int(d / 2)
    new_f = filename + '-moved'
    with open(filename, 'r') as in_f, open(new_f, 'w') as out_f:
        for line in in_f.readlines():
            prts = line.split(',')
            prts = [prt.strip() for prt in prts]
            line = ''
            for x in prts[:-1]:
                line += str(int(x) - m) + ','
            line += prts[-1] + '\n'
            out_f.write(line)
    return new_f


def generate_data(routes_lines, routes_stops, hours_n, hours_buses, trips_m, trips_transfers, should_generate):
    import generate_hours, generate_trips, generate_train_test_data
    import util

    if should_generate:
        print('Generating new data')
        generate_hours.generate_hours(hours_buses, hours_n)
        generate_trips.generate_trips(routes_lines, routes_stops, trips_m, trips_transfers)
        generate_train_test_data.generate_test_train_data_without_filenames(routes_lines, routes_stops, trips_m,
                                                                            trips_transfers, hours_n, hours_buses)

    train_data_filename = util.get_train_data_filename(routes_lines, routes_stops, trips_m, trips_transfers, hours_n,
                                                       hours_buses)
    test_data_filename = util.get_test_data_filename(routes_lines, routes_stops, trips_m, trips_transfers, hours_n,
                                                     hours_buses)
    # TODO move scaling data somewhere else.
    from util import DAY_LENGTH
    train_data_filename = move_data(train_data_filename, DAY_LENGTH)
    test_data_filename = move_data(test_data_filename, DAY_LENGTH)

    return train_data_filename, test_data_filename


def get_nn_filenames(routes_lines, routes_stops, hours_n, hours_buses, trips_m, trips_transfers, nn):
    import util
    savefig_filename = util.get_savefig_filename(routes_lines, routes_stops, trips_m, trips_transfers, hours_n,
                                                 hours_buses, nn)
    save_nn_filename = util.get_save_nn_filename(routes_lines, routes_stops, trips_m, trips_transfers, hours_n,
                                                 hours_buses, nn)
    return savefig_filename, save_nn_filename


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Neural Network framework.')
    parser.add_argument('-g', '--generate', action='store_true',
                        help='If specified new datafiles will be generated. Otherwise using existing files')
    parser.add_argument('-s', '--save', action='store_true',
                        help='If specified results will be saved to files')

    args = parser.parse_args()
    should_generate = args.generate

    from datetime import datetime    
    import random
    random.seed(123)
    import config
    routes_lines, routes_stops, hours_n, hours_buses, trips_m, trips_transfers = config.get_filenames_params()
    train_data_filename, test_data_filename = generate_data(routes_lines, routes_stops, hours_n, hours_buses,
                                                            trips_m, trips_transfers, should_generate)
    from perceptron import genetic
    date_train_neat = datetime.now()#training
    best_net = genetic.main(train_data_filename, test_data_filename, neat_params=config.get_neat_params(),
                 n_generations=config.n_generations, train_params=config.get_train_params(), save=args.save)
        
    date_test_neat = datetime.now()#testing neat
    from test_neat import test_neat
    test_neat_result  = test_neat(best_net)
    
    date_test_monte_carlo_small = datetime.now()#testing simulator- monte carlo
    number_of_tests = config.monte_carlo_tests_small
    from test_monte_carlo import test_monte_carlo
    test_monte_carlo_result_small  = test_monte_carlo(number_of_tests)

    date_test_monte_carlo_medium = datetime.now()#testing simulator- monte carlo
    number_of_tests = config.monte_carlo_tests_medium
    from test_monte_carlo import test_monte_carlo
    test_monte_carlo_result_medium  = test_monte_carlo(number_of_tests)


    date_test_monte_carlo_big = datetime.now()#testing simulator- monte carlo
    number_of_tests = config.monte_carlo_tests_big
    from test_monte_carlo import test_monte_carlo
    test_monte_carlo_result_big  = test_monte_carlo(number_of_tests)

    date_end = datetime.now()#end

    print("\n\n")
    print("Training neat:"+str(date_test_neat-date_train_neat))
    print("Testing neat:"+str(date_test_monte_carlo_small-date_test_neat))
    print("Testing small monte carlo simulator:"+str(date_test_monte_carlo_medium-date_test_monte_carlo_small))
    print("Testing medium monte carlo simulator:"+str(date_test_monte_carlo_medium-date_test_monte_carlo_small))
    print("Testing big monte carlo simulator:"+str(date_end -date_test_monte_carlo_big))

    print("Result neat:")
    print(test_neat_result)
    print("Result small monte carlo simulator:")
    print(test_monte_carlo_result_small)
    print("Result medium monte carlo simulator:")
    print(test_monte_carlo_result_medium)
    print("Result big monte carlo simulator:")
    print(test_monte_carlo_result_big)



if __name__ == "__main__":
    main()
