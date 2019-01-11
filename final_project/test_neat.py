#montecarlo +  neural network

def getKey(item):
    return item[1]


def test_neat(neural_network):
    import config
    import util
    import random
    from perceptron.util import scale_data
    routes_lines, routes_stops, hours_n, hours_buses, trips_m, trips_transfers = config.get_filenames_params()
    number_of_tests = config.neat_tests
    tests =[]

    random.seed(123)


    hours_filename2 = util.get_hours_results_filename(number_of_tests, hours_buses)
    with open(hours_filename2, 'w') as file:
        for i in range(number_of_tests):
            hours = []
            hours2 = []
            for bus in hours_buses:
                line_hours = []
                for _ in range(bus):
                    value = random.randint(0, util.DAY_LENGTH)
                    hours.append(value - util.DAY_LENGTH/2 )
                    line_hours.append(value)
                hours2.append(line_hours)

            tests.append(hours)
            file.write(str(hours2))
            file.write('\n')



    results =  neural_network.test_without_y(tests)
    results.sort(key = getKey)

    #print(results)
    #print("\n")
    bests = int(number_of_tests/10)
    best_results = results[:bests]
    #print(best_results)
    routes_filename = util.get_routes_filename(routes_lines, routes_stops)
    trips_filename = util.get_trips_filename(routes_lines, routes_stops, trips_m, trips_transfers)
    hours_filename = util.get_hours_results_filename(hours_n, hours_buses) 

    allhours = []
    with open(hours_filename, 'w') as file:
        for i in range(bests):
            hours = []
            actual_hours = best_results[i][0]
            j = 0
            for bus in hours_buses:
                line_hours = []
                for _ in range(bus):
                    line_hours.append(int(actual_hours[j]+ util.DAY_LENGTH/2))
                    j = j+1
                hours.append(line_hours)
            allhours.append(hours)
            file.write(str(hours))
            file.write('\n')
    
    import simulator

    results_checked = simulator.run_simulator(routes_filename, trips_filename, hours_filename)
    #print(tests)
    #print(results_checked)

    results_checked, _ = scale_data(results_checked,[0])
    
    pairs_resultNeat_resultSimulator = zip(results_checked, best_results)

    best_result = min(results_checked)
    index_best_result = results_checked.index(best_result)
    res = list(zip(best_results, results_checked))

    final_best_result = ((allhours[index_best_result], best_results[index_best_result][1]), best_result)

    

    return res, final_best_result, hours_filename2, pairs_resultNeat_resultSimulator

def calculate_error(result_pairs):
    import config
    error = 0
    for sim_value, neat_value in result_pairs:
        error = error + abs(sim_value-neat_value[1])/sim_value

    
    
    return error,  100* error/(config.neat_tests/10)






