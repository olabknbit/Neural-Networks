def generate_data(routes_lines, routes_stops, hours_n, hours_buses, trips_m, trips_transfers, nn, number_of_epochs):
    import generate_hours, generate_trips, generate_train_test_data, util
    from perceptron import regression
    generate_hours.generate_hours(hours_buses, hours_n)
    generate_trips.generate_trips(routes_lines, routes_stops, trips_m, trips_transfers)
    generate_train_test_data.generate_test_train_data_without_filenames(routes_lines, routes_stops, trips_m,
                                                                        trips_transfers, hours_n, hours_buses)

    train_data_filename = util.get_train_data_filename(routes_lines, routes_stops, trips_m, trips_transfers, hours_n,
                                                       hours_buses)
    test_data_filename = util.get_test_data_filename(routes_lines, routes_stops, trips_m, trips_transfers, hours_n,
                                                     hours_buses)
    savefig_filename = util.get_savefig_filename(routes_lines, routes_stops, trips_m, trips_transfers, hours_n,
                                                 hours_buses)
    save_nn_filename = util.get_save_nn_filename(routes_lines, routes_stops, trips_m, trips_transfers, hours_n,
                                                 hours_buses)
    from perceptron.util import tanh, tanh_derivative
    accuracy = regression.main(train_data_filename, test_data_filename, activation_f=tanh,
                               activation_f_derivative=tanh_derivative, visualize_every=None,
                               create_nn=nn, save_nn=save_nn_filename, read_nn=None, number_of_epochs=number_of_epochs,
                               l_rate=0.001, biases=True, savefig_filename=savefig_filename)
    print('accuracy', accuracy)


def main():
    nn = [3] # nn model
    number_of_epochs = 1000
    for routes_lines in [1]:
        for routes_stops in [6]:
            for hours_n in [1000]: # how many rows of train and test data to generate (split 60:40)
                for hours_buses in [[1]]: # how many buses of each line should run
                    for trips_m in [10]: # how many passenger's trips there are
                        for trips_transfers in [0]: # max how many transfers each passenger can have
                            generate_data(routes_lines, routes_stops, hours_n, hours_buses, trips_m, trips_transfers,
                                          nn, number_of_epochs)


if __name__ == "__main__":
    main()
