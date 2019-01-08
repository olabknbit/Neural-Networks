def test_hours_file(file_name):
    import config
    import util
    import random
    routes_lines, routes_stops, hours_n, hours_buses, trips_m, trips_transfers = config.get_filenames_params()


    routes_filename = util.get_routes_filename(routes_lines, routes_stops)
    trips_filename = util.get_trips_filename(routes_lines, routes_stops, trips_m, trips_transfers)


    import simulator
    results = simulator.run_simulator(routes_filename, trips_filename, file_name)

    best_result = min(results)
    index_best_result = results.index(best_result)

    from perceptron.util import scale_data
    results_checked, _ = scale_data([best_result],[0])
    

    final_result = ([],results_checked)
    return final_result