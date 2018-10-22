def main():
    import classification

    prefix = 'projekt1/classification/data'
    modes = ['simple', 'three_gauss']
    quantities = [100, 500, 1000, 10000]

    create_nn = [[], [7], [6, 7, 8]]

    for mode in modes:
        for quantity in quantities:
            def get_filename(t):
                return prefix + '.' + mode + '.' + t + '.' + str(quantity) + '.csv'

            train_filename = get_filename('train')
            test_filename = get_filename('test')
            n_epochs = 10000
            for nn in create_nn:
                save_nn = 'classification.' + mode + '.' + str(quantity) + str(nn)
                classification.main(train_filename, test_filename, nn, save_nn, None, n_epochs, n_epochs, 0.001, True)


if __name__ == "__main__":
    main()
