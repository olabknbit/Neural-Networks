def plot_data(X_train, y_train, X_test, y_test, y_predicted, y_sklearn_predicted=None, savefig_filename=None):
    if len(X_train[0]) != 1:
        print('Cannot plot because len(data) = %d > 1' % len(X_train[0]))
        return
    import matplotlib.pyplot as plt
    marker = '.'
    plt.clf()

    plt.plot(X_test, y_test, marker, c='red', label='test_data')
    plt.plot(X_test, y_predicted, marker, c='blue', label='predicted')
    plt.plot(X_train, y_train, marker, c='green', label='training_data')
    if y_sklearn_predicted is not None:
        plt.plot(X_test, y_sklearn_predicted, marker, c='purple', label='sklearn_predicted')

    plt.legend()
    if savefig_filename is not None:
        plt.savefig(savefig_filename)


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

    AvgError = 1.0 * Error / (1.0 * len(y_test))
    MSE = 1.0 * SE / (1.0 * len(y_test))
    RMSE = MSE ** 0.5
    RAE = Error / RAEDivider
    RRAE = SE / RRAEDivider

    print('Error = %.3f' % Error)
    print('AvgError = %.3f' % AvgError)
    print('MSE = %.3f' % MSE)
    print('RMSE = %.3f' % RMSE)
    print('RAE = %.3f' % RAE)
    print('RRAE = %.3f' % RRAE)


def get_nn(create_nn, read_nn, biases, activation_f, activation_f_derivative, X_train):
    from perceptron.util  import read_network_from_file, initialize_network
    neural_network = None
    if create_nn is not None:
        # Calculate the number of inputs and outputs from the data.
        n_inputs = len(X_train[0])
        neural_network = initialize_network(create_nn, n_inputs, biases, activation_f, activation_f_derivative)
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


def main(train_filename, test_filename, create_nn, save_nn, read_nn, number_of_epochs, visualize_every, l_rate, biases,
         savefig_filename, activation_f, activation_f_derivative, compare_to_sklearn=False):
    if train_filename is None or test_filename is None:
        print('Both train and test filename has to be provided for scaling')
        exit(1)

    from perceptron.util  import get_split_dataset
    X_train, y_train, X_test, y_test = get_split_dataset(train_filename, test_filename)
    neural_network = get_nn(create_nn, read_nn, biases, activation_f, activation_f_derivative, X_train)

    # Train neural network.
    from perceptron.neural_network_regression import train
    train(neural_network, X_train, y_train, number_of_epochs, l_rate, visualize_every)

    if save_nn is not None:
        from perceptron.util import write_network_to_file
        write_network_to_file(save_nn, neural_network)

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
        plot_data(X_train, y_train, X_test, y_test, y_predicted, y_sklearn_predicted, savefig_filename)
    return avg_error
