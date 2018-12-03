PREFIX = 'final_project/data/'
ROUTES_FILENAME = 'routes-3-lines-6-stops.txt'
HOURS_FILENAME = 'hours-3-lines-6-stops-not-for-training.txt'
TRIPS_FILENAME = 'trips-3-lines-6-stops-6-M-2-transfers.txt'


def main():
    import simulator
    results = simulator.run_simulator(PREFIX + ROUTES_FILENAME, PREFIX + TRIPS_FILENAME, PREFIX + HOURS_FILENAME)
    print(results)


if __name__ == "__main__":
    main()
