def main():
    prefix = 'final_project/'
    stops = dict()
    lines = []
    first_stops = dict()
    with open(prefix + 'routes.txt', 'r') as file:
        routes = file.readlines()

        for route in routes:
            str_splt = map(str.strip, route.split(','))
            line = dict()
            line_id = str_splt[0]
            first_stop = str_splt[2]

            first_stops[line_id] = first_stop

            lines.append(line_id)

            i = 2
            while i < len(str_splt) - 2:
                stop_id = str_splt[i]
                t = int(str_splt[i + 1])
                next_stop_id = str_splt[i + 2]
                line[stop_id] = (t, next_stop_id)
                i += 2

            stops[line_id] = line

    with open(prefix + 'hours.txt', 'r') as file:
        hours_l = file.readlines()
        hours_l = map(eval, hours_l)

    with open(prefix + 'trips.txt', 'r') as file:
        trips = file.readlines()

        passenger_trips = []
        for trip in trips:
            str_splt = map(str.strip, trip.split(','))

            start_time = int(str_splt[0])
            first_stop = str_splt[1]

            transfers = dict()

            i = 1
            while i < len(str_splt) - 2:
                stop_id = str_splt[i]
                line_id = str_splt[i + 1]
                next_stop_id = str_splt[i + 2]
                transfers[stop_id] = (line_id, next_stop_id)
                i += 2

            passenger_trips.append((start_time, first_stop, transfers))

    if routes is None or hours_l is None or trips is None:
        print('Failed to read buses info or hours info or trips info', routes, hours_l, trips)
        exit(1)

    import simulator
    results = simulator.run(stops, lines, first_stops, hours_l, passenger_trips)
    print(results)


if __name__ == "__main__":
    main()