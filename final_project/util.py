def get_routes_parsed_info(routes_filename):
    stops = dict()
    lines = []
    first_stops = dict()
    with open(routes_filename, 'r') as file:
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
    return routes, stops, lines, first_stops
