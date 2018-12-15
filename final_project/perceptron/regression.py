def print_data(y_test, predicted_outputs):
    Error = 0.0
    SE = 0.0
    RAEDivider = 0.0
    RRAEDivider = 0.0
    AvgData = sum(predicted_outputs) / len(predicted_outputs)

    for y, pred in zip(y_test, predicted_outputs):
        Error += abs(pred - y)
        SE += (pred - y) ** 2
        RAEDivider += abs(AvgData - y)
        RRAEDivider += (AvgData - y) ** 2

    AvgError = Error / float(len(y_test))
    MSE = SE / float(len(y_test))
    RMSE = MSE ** 0.5
    RAE = Error / RAEDivider
    RRAE = SE / RRAEDivider

    print('Error = %.3f' % Error)
    print('AvgError = %.3f' % AvgError)
    print('MSE = %.3f' % MSE)
    print('RMSE = %.3f' % RMSE)
    print('RAE = %.3f' % RAE)
    print('RRAE = %.3f' % RRAE)


def get_nn(create_nn, read_nn, activation_f, activation_f_derivative, X_train):
    from util import read_network_from_file, initialize_network
    neural_network = None
    if create_nn is not None:
        # Calculate the number of inputs and outputs from the data.
        n_inputs = len(X_train[0])
        neural_network = initialize_network(n_inputs, activation_f, activation_f_derivative)
    elif read_nn is not None:
        neural_network = read_network_from_file(read_nn, activation_f, activation_f_derivative)

    else:
        print('Cannot create nor read nn. Exiting')
        exit(1)

    return neural_network


def sklearn_test(nn, X_train, y_train, X_test, y_test):
    from sklearn.neural_network import MLPRegressor
    lr = MLPRegressor(hidden_layer_sizes=nn, activation='tanh').fit(X_train, y_train)
    return lr.predict(X_test)


def main(train_filename, test_filename, create_nn, save_nn, read_nn, number_of_epochs, visualize_every, l_rate,
         savefig_filename, activation_f, activation_f_derivative, compare_to_sklearn=False):
    if train_filename is None or test_filename is None:
        print('Both train and test filename has to be provided for scaling')
        exit(1)

    from util import get_split_dataset
    X_train, y_train, X_test, y_test = get_split_dataset(train_filename, test_filename)
    neural_network = get_nn(create_nn, read_nn, activation_f, activation_f_derivative, X_train)

    # Train neural network.
    neural_network.train(X_train, y_train, number_of_epochs, l_rate, visualize_every)

    if save_nn is not None:
        from util import write_network_to_file_regression
        write_network_to_file_regression(save_nn, neural_network)

    # Test the neural network.
    avg_error, y_predicted = neural_network.test(X_test, y_test)
    print('NN accuracy and errors')
    print_data(y_test, y_predicted)

    y_sklearn_predicted = None
    if compare_to_sklearn:
        y_sklearn_predicted = sklearn_test(create_nn, X_train, y_train, X_test, y_test)
        print('\nsklearn accuracy and errors')
        print_data(y_test, y_sklearn_predicted)

    if savefig_filename is not None:
        from util import plot_regression_data
        plot_regression_data(X_train, y_train, X_test, y_test, y_predicted, y_sklearn_predicted, savefig_filename)
    return avg_error
