# file names params
routes_lines = 4
routes_stops = 6
hours_n = 4000 # how many rows of train and test data to generate (split 60:40)
hours_buses = [2,5,3,4]  # how many buses of each line should run
trips_transfers = 2  # max how many transfers each passenger can have
trips_m = 700  # how many passenger's trips there are

# NEAT params
COMPATIBILITY_THRESHOLD = 2.5
FITTEST_PERCENTAGE = 0.7
BABIES_PER_GENERATION = 2
ASEXUAL_REPRODUCTION_CHANCE = 0.25
ADD_SYNAPSE_MUTATION_CHANCE = 0.5
ADD_NEURON_MUTATION_CHANCE = 0.3

# train params
n_iter = 1000
l_rate = 0.001
n_generations = 10

# test params
neat_tests = 5000
monte_carlo_tests_small = 500
monte_carlo_tests_medium = 10
monte_carlo_tests_big= 10000

# alg params
no_avaible_bus_penalty = 1000 


def get_filenames_params():
    return routes_lines, routes_stops, hours_n, hours_buses, trips_m, trips_transfers


def get_neat_params():
    return COMPATIBILITY_THRESHOLD, FITTEST_PERCENTAGE, BABIES_PER_GENERATION, ASEXUAL_REPRODUCTION_CHANCE, \
           ADD_SYNAPSE_MUTATION_CHANCE, ADD_NEURON_MUTATION_CHANCE


def get_train_params():
    return n_iter, l_rate

def get_test_params():
    return neat_tests, monte_carlo_tests_small, monte_carlo_tests_medium, monte_carlo_tests_big, n_generations

