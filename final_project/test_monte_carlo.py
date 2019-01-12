#montecarlo +  simulator


def test_monte_carlo(number_of_tests):
    import config
    import util
    import random
    routes_lines, routes_stops, hours_n, hours_buses, trips_m, trips_transfers = config.get_filenames_params()

    random.seed(234)

    routes_filename = util.get_routes_filename(routes_lines, routes_stops)
    trips_filename = util.get_trips_filename(routes_lines, routes_stops, trips_m, trips_transfers)
    hours_filename = util.get_hours_filename(number_of_tests, hours_buses)

    tests =[]
    # for i in range(number_of_tests):
    #     hours = []
    #     for bus in hours_buses:
    #         for _ in range(bus):
    #             hours.append(random.randint(-util.DAY_LENGTH/2, util.DAY_LENGTH/2))
    #     tests.append(hours)

    # with open(hours_filename, 'w') as file:
    #     for _ in range(tests):
    #         hours = []
    #         for bus in hours_buses:
    #             line_hours = best_results[i][0]
    #             i = i+1
    #             hours.append(line_hours)
    #         file.write(str(hours))
    #         file.write('\n')
    with open(hours_filename, 'w') as file:
        for _ in range(number_of_tests):
            hours = []
            for bus in hours_buses:
                line_hours = []
                for _ in range(bus):
                    line_hours.append(random.randint(0, util.DAY_LENGTH))
                line_hours.sort()
                hours.append(line_hours)#Sortowanie
            tests.append(hours)
            file.write(str(hours))
            file.write('\n')


    import simulator
    results = simulator.run_simulator(routes_filename, trips_filename, hours_filename)

    best_result = min(results)
    index_best_result = results.index(best_result)

    from perceptron.util import scale_data
    results_checked, _ = scale_data([best_result],[0])
    

    final_result = (tests[index_best_result], results_checked)
    return final_result