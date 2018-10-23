def main():
    import classification

    prefix = 'projekt1/classification/data'
    modes = ['simple']
    quantities = [500]

    create_nn = [[], [7], [6, 7, 8]]

    for mode in modes:
        for quantity in quantities:
            def get_filename(t):
                return prefix + '.' + mode + '.' + t + '.' + str(quantity) + '.csv'

            train_filename = get_filename('train')
            test_filename = get_filename('test')
            n_epochs = 10000
            for nn in create_nn:
                save_nn = 'reLu/classification.' + mode + '.' + str(quantity) + str(nn)
                from util import sigmoid, sigmoid_derivative, reLu, reLu_derivative
                # activation_f, activation_f_derivative = sigmoid, sigmoid_derivative
                activation_f, activation_f_derivative = reLu, reLu_derivative
                classification.main(train_filename, test_filename, nn, save_nn, None, n_epochs, n_epochs / 10, 0.001,
                                    True, activation_f, activation_f_derivative)


if __name__ == "__main__":
    main()
