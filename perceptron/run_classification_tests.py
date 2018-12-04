def main():
    import classification
    from util import reLu, reLu_derivative

    prefix = 'projekt1-oddanie/clasification/data'
    modes = ['circles']
    quantities = [500, 1000]

    create_nn = [[1000]]

    # (sigmoid, sigmoid_derivative, 'sigmoid'),
    activation = [(reLu, reLu_derivative, 'reLu')]

    for mode in modes:
        for quantity in quantities:
            def get_filename(t):
                return prefix + '.' + mode + '.' + t + '.' + str(quantity) + '.csv'

            train_filename = get_filename('train')
            test_filename = get_filename('test')
            n_epochs = 10000
            for nn in create_nn:
                for ff in activation:
                    activation_f, activation_f_derivative, name_f = ff
                    save_nn = name_f + '1-oddanie/classification.' + mode + '.' + str(quantity) + str(nn)
                    print(save_nn)

                    # activation_f, activation_f_derivative = reLu, reLu_derivative
                    classification.main(train_filename, test_filename, nn, save_nn, None, n_epochs, n_epochs / 10, 0.001,
                                        True, activation_f, activation_f_derivative)


if __name__ == "__main__":
    main()
