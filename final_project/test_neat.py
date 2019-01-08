#montecarlo +  neural network

def getKey(item):
    return item[1]


def test_neat(neural_network):
    import config
    import util
    import random
    routes_lines, routes_stops, hours_n, hours_buses, trips_m, trips_transfers = config.get_filenames_params()
    number_of_tests = config.neat_tests
    tests =[]

    for i in range(number_of_tests):
        hours = []
        for bus in hours_buses:
            for _ in range(bus):
                hours.append(random.randint(-util.DAY_LENGTH/2, util.DAY_LENGTH/2))
        tests.append(hours)


    results =  neural_network.test_without_y(tests)
    results.sort(key = getKey, reverse = False)

    #print(results)

    bests = int(number_of_tests/10)
    best_results = results[:bests]
    #for i in best_results:
    #    print(i)
    #check results

    routes_filename = util.get_routes_filename(routes_lines, routes_stops)
    trips_filename = util.get_trips_filename(routes_lines, routes_stops, trips_m, trips_transfers)
    hours_filename = util.get_hours_results_filename(hours_n, hours_buses) 

    i = 0
    # with open(hours_filename, 'w') as file:
    #     for _, hours in enumerate(best_results):
    #         print(hours[0])
    #         file.write(str(hours[0]))
    #         file.write('\n')

    with open(hours_filename, 'w') as file:
        for i in range(bests):
            hours = []
           
            for bus in hours_buses:
                k=0
                line_hours = []
                for j in range(bus):
                    line_hours.append(best_results[i][0][bus+k])
                    k=k+1
                hours.append(line_hours)

            file.write(str(hours))
            file.write('\n')
    
    import simulator

    results_checked = simulator.run_simulator(routes_filename, trips_filename, hours_filename)

    from perceptron.util import scale_data
    results_checked, _ = scale_data(results_checked,[0])
    res = list(zip(best_results, results_checked))
    return res






