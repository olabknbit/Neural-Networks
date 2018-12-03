def get_times_at_stops(stops_orig, lines, first_stops, hours):
    # create a dict of a form: key = line_id; value another dict where key = stop_id, value = list of times when buses
    # of that line stop at this stop
    buses_at_stops = dict()
    for hour, line in zip(hours, lines):
        bas = dict()
        stop = first_stops[line]
        stops = stops_orig[line]

        t = 0
        while stop in stops.keys():
            ss = []
            for h in hour:
                ss.append(h + t)
            bas[stop] = ss
            t_new, s_new = stops[stop]
            stop = s_new
            t += t_new
        buses_at_stops[line] = bas

    return buses_at_stops


def get_next_bus_time(times, start_time):
    for time in times:
        if time >= start_time:
            return time
    return None


def get_time_at_next_stop(buses_at_stops, stops_orig, line_id, stop, start_time, final_stop):
    times = buses_at_stops[line_id][stop]
    nbt = get_next_bus_time(times, start_time)
    if nbt is None:
        return None

    start_time = nbt
    ttg, ns = stops_orig[line_id][stop]
    start_time += ttg
    if ns == final_stop:
        return start_time

    return get_time_at_next_stop(buses_at_stops, stops_orig, line_id, ns, start_time, final_stop)


def run(stops_orig, lines, first_stops, hours_l, trips):
    results = []
    for hours in hours_l:
        buses_at_stops = get_times_at_stops(stops_orig, lines, first_stops, hours)
        overall_t = 0
        for start_time, stop, transfers in trips:
            t = start_time

            while stop in transfers.keys():
                line_id, next_stop = transfers[stop]
                start_time = get_time_at_next_stop(buses_at_stops, stops_orig, line_id, stop, start_time, next_stop)
                if start_time is None:
                    # penalty bc someone just got stranded at the bus stop.
                    start_time = 100000
                    break
                stop = next_stop

            overall_t += start_time - t
        results.append(overall_t/len(trips))
    return results


def run_simulator(routes_filename, trips_filename, hours_filename):
    import util
    routes, stops, lines, first_stops = util.get_routes_parsed_info(routes_filename)

    with open(hours_filename, 'r') as file:
        hours_l = file.readlines()
        hours_l = map(eval, hours_l)

    with open(trips_filename, 'r') as file:
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

    return run(stops, lines, first_stops, hours_l, passenger_trips)
