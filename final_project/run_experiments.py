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
    best_net, nets_score = genetic.main(train_data_filename, test_data_filename, neat_params=config.get_neat_params(),
                 n_generations=config.n_generations, train_params=config.get_train_params(), save=args.save)
        
    date_test_neat = datetime.now()#testing neat
    from test_neat import test_neat
    test_neat_result, test_neat_result_single, generated_hours_file, pairs_result = test_neat(best_net)
    
    date_test_generated_hours = datetime.now()#testing hours generated for neat
    # from test_hours_file import test_hours_file # bez testowania całego zbioru testowego
    # test_generated_hours_result= test_hours_file(generated_hours_file)

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

    from test_neat import calculate_error
    error, error_percent = calculate_error(pairs_result)



    string_result = ("Training neat: "+str(date_test_neat-date_train_neat) +
        "\nTesting neat: "+str(date_test_generated_hours-date_test_neat)+

        "\nTesting neat hours: "+str(date_test_monte_carlo_small-date_test_generated_hours)+
        "\nTesting small monte carlo simulator: "+str(date_test_monte_carlo_medium-date_test_monte_carlo_small)+
        "\nTesting medium monte carlo simulator: "+str(date_test_monte_carlo_big-date_test_monte_carlo_medium)+
        "\nTesting big monte carlo simulator: "+str(date_end -date_test_monte_carlo_big)+


        "\n\nResult single neat: " + str(test_neat_result_single)+
        "\nNeat error: " + str(error)+
        "\nNeat error_percent: " + str(error_percent)+
        # "\nResult single generated_hours: " + str(test_generated_hours_result)+
        "\nResult small monte carlo simulator: "+ str(test_monte_carlo_result_small)+
        "\nResult medium monte carlo simulator: "+ str(test_monte_carlo_result_medium)+
        "\nResult big monte carlo simulator: " + str(test_monte_carlo_result_big)+

        "\n\nResult single neat/big: " + 
        str(100*test_neat_result_single[1]/test_monte_carlo_result_big[1][0])+
        # "\nResult genrated hours /big: " + 
        # str(100*test_generated_hours_result[1][0]/test_monte_carlo_result_big[1][0])+
        "\nResult small monte carlo simulator/big: " + 
        str(100*test_monte_carlo_result_small[1][0]/test_monte_carlo_result_big[1][0])+
        "\nResult medium monte carlo simulator/big: "+ 
        str(100*test_monte_carlo_result_medium[1][0]/test_monte_carlo_result_big[1][0]))
   
    print(string_result)

    directory = 'results/' + str(date_end.strftime('%H-%M-%S-%d-%m-%Y')) + '/' 

    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_to_save_result_name = directory + "results.txt"
    with open(file_to_save_result_name, 'w') as file:
        file.write(string_result)

    file_to_save_values_name = directory + "values.txt"
    with open(file_to_save_values_name, 'w') as file:
        for i in test_neat_result:
             file.write(str(i))

    file_to_save_neat_scores_name = directory + "neat_scores.txt"
    with open(file_to_save_neat_scores_name, 'w') as file:
        for i in nets_score:
            file.write(str(i))
            file.write('\n')

    file_to_save_config_name = directory + "config.txt"
    with open(file_to_save_config_name, 'w') as file:
        file.write("Filename params: \n"+str(config.get_filenames_params()))
        file.write("\nNeat params: \n"+str(config.get_neat_params()))
        file.write("\nTrain params: \n"+str(config.get_train_params()))
        file.write("\nTest params: \n"+str(config.get_test_params()))

    from perceptron.util import visualize_result
    visualize_result(nets_score, directory + "neat_scores.png")


    # print("Result neat:")
    # print(test_neat_result)

if __name__ == "__main__":
    main()
