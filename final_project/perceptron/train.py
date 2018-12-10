def train(network, X_train, y_train, l_rate, n_iter, visualize_every):
    import numpy as np
    last_error = - np.infty
    for epoch in range(n_iter):
        iter_error = 0.0
        for row, expected in zip(X_train, y_train):
            # The net should only predict the class based on the features,
            # so the last cell which represents the class is not passed forward.
            output = network.forward_propagate(row)[0]

            iter_error += np.sqrt((expected - output) ** 2)
            network.backward_propagate(expected)
            network.update_weights(row, l_rate)
        if visualize_every is not None and epoch % visualize_every == 0:
            import visualize
            visualize.main(network, epoch)

        if epoch % 100 == 0:
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, iter_error))
            # Stop training if iter_error not changing.
            # TODO consider stochastic batches.
            if abs(last_error - iter_error) < 0.001:
                break
            last_error = iter_error