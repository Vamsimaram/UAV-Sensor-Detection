import json
import math
import importlib.util

from pyproj import Geod
from shapely.geometry import Point, Polygon
from pyproj import Transformer
from shapely.ops import transform

# spec = importlib.util.spec_from_file_location("dgal", "/Users/alexbrodsky/Documents/OneDrive - George Mason University - O365 Production/aaa_python_code/aaa_dgalPy/lib/dgalPy.py")
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
dgal_file_path = os.path.join(current_dir, "aaa_lib_dgalPy", "dgalPy.py")
spec = importlib.util.spec_from_file_location("dgal", dgal_file_path)
dgal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dgal)
#------------------------------------------
#  TBD
#
# - *** compute (long,lat) of verticies for protected areas for Barreto's display
# - *** ! update distance function from euclidean distance to distance between two (long,lat) points
# - *** try Pedro's data
# - *** check input; including that the squares in area_of_interest include reaction_time & detection areas
# - *** add user_view w/long/lat coordinates for vertices of target areas; also, give actual locations for sensors w/long-lat
# - *** wrap-up a couple of optimization template + predictive
# - *** add user_view function, that computes additional metrics for user display, incl. (long,lat) coords of vertices of protected areas
# - *** try Pedro's generated preprocessing
# - add meta-optimization alg for maximizing reaction distance, by running cost minimization multiple times and
#   and (logarithmically) converging to max reaction time for which the problem is still feasible,
#   and Pd before reaction time is within desirable limits
# - install and try running problems using CPLEX
# - add DG-ViTh requirement spec optimization on top of sensor placement problem
# - add classification probability per square as input; compute metrics similar to those for
#   detection.
# -
#------------------------------------------
# import ../aaa_dgalPy/lib/dgalPy.py as dgal
# import /Users/alexbrodsky/Documents/OneDrive\ -\ George\ Mason\ University\ -\ O365\ Production/aaa_python_code/aaa_dgalPy/lib/dgalPy.py as dgal

def out(v):
    print("\n v=",v)
    return True

# for Boolean a,b
def iff(a,b):
    return (a == b)

# convert degrees to radians
def d2r(alpha):
    return (alpha * math.pi / 180)

# convert radians to degrees
def r2d(alpha):
    return (alpha * 180 / math.pi)

# ------------------------------------------------------------------------------
# the dist function computes the distance to travel from (lon1,lat1) to (lon2,lat2)
#def dist(lon1,lat1,lon2,lat2, eRadius):
#    lon1r = d2r(lon1)
#    lat1r = d2r(lat1)
#    lon2r = d2r(lon2)
#    lat2r = d2r(lat2)
#    #angleInRad = math.acos( math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2-lon1) )
#    angleInRad = math.acos(math.sin(lat1r) * math.sin(lat2r) + math.cos(lat1r) * math.cos(lat2r) * math.cos(lon2r - lon1r)) # corrected the above line
#    distance = angleInRad * eRadius
#    return distance
# ----------------------------------
# Replaced the above function using the pyproj library
# Calculate the geo distance between two points
def dist(lon1,lat1,lon2,lat2):
    # Define the WGS84 ellipsoid
    wgs84_geod = Geod(ellps="WGS84")

    angle1, angle2, distance = wgs84_geod.inv(lon1, lat1, lon2, lat2)
    # Return the distance in kilometers
    return distance / 1000

# ------------------------------------------------------------------------------
# the following computes new (lon2,lat2) when moving DIST from (lon1,lat1) along AZIMUTH
# azimuth, lon1, lat1 in degrees; dist & eRadius in the same distance units
#def longLat(lon1, lat1, azimuth, dist, eRadius):
#    lon1r = d2r(lon1)
#    lat1r = d2r(lat1)
#    azR = d2r(azimuth)
#    aDist = dist / eRadius
#    lat2r = math.asin( math.sin(lat1r) * math.cos(aDist) + math.cos(lat1r) * math.sin(aDist) * math.cos(azR) )
#    lon2r = lon1r + \
#            math.atan2( math.sin(azR) * math.sin(aDist) * math.cos(lat1r), \
#                        math.cos(aDist) - math.sin(lat1r) * math.sin(lat2r)
#
#            )
#    lon2 = r2d(lon2r)
#    lat2 = r2d(lat2r)
#    return [ lon2, lat2 ]
# --------------------------------------
# Replaced the above function using the pyproj library
# Calculate the end point given the start point, azimuth, and distance in kilometers
def longLat(lon1, lat1, azimuth, dist):
    # Define the WGS84 ellipsoid
    wgs84_geod = Geod(ellps="WGS84")

    lon2, lat2, back_azimuth = wgs84_geod.fwd(lon1, lat1, azimuth, dist * 1000)

    return [ lon2, lat2 ]

#-------------------------------------------------------------------------------
# debug; must be implemented in LP style
# non LP implmentation of flag functions, for debugging

# def flagIffXgeZero(f, x):
#     a = (f == 1)
#     b = (x - eps >= 0)
#     return iff(a,b)

#------------------------
def valid_am_input(input):
    return True
    # needs to be written to express that all lengths fit etc
    # in particular, check that all 0 <= Pd < 1 (never 1, or else ldPnd is not defined)
    # in fact, put upper bound on Pd so that lgPnd >= -15

#-------------------------------------------------------------------------------
# Generate lon,lat corner vertices for each square in the area_of_interest
def generate_lon_lat_vertices(area_of_interest, config):
    square_side = config["square_side"]
    sw_corner = config["south_west_corner"]

    vertices_list = []

    for coord in area_of_interest:
        x_index, y_index = coord
        # Calculate southwest corner longitude and latitude
        sw_long = sw_corner["long"] + (x_index * square_side)
        sw_lat = sw_corner["lat"] + (y_index * square_side)

        # Calculate the other three corners based on square side
        nw_long = sw_long
        nw_lat = sw_lat + square_side
        ne_long = sw_long + square_side
        ne_lat = sw_lat + square_side
        se_long = sw_long + square_side
        se_lat = sw_lat

        # Append the coordinates of the four corners for the current square
        square_vertices = {
            "SW": {"long": sw_long, "lat": sw_lat},
            "NW": {"long": nw_long, "lat": nw_lat},
            "NE": {"long": ne_long, "lat": ne_lat},
            "SE": {"long": se_long, "lat": se_lat}
        }
        vertices_list.append(square_vertices)

    return vertices_list
#-------------------------------------------------------------------------------
# Calculate the surface area of a square or polygon given its corner vertices in
# longitude and latitude degrees, considering the Earth's curvature
def calculate_polygon_area(vertices):
    """
    vertices: A list of tuples, where each tuple contains the longitude and latitude
                of a vertex in degrees. The vertices must define a closed loop, where
                the first and last vertex are the same.
    """
    # WGS84 ellipsoid - a commonly used geodetic reference system
    geod = Geod(ellps='WGS84')

    lon, lat = zip(*vertices)

    area, _ = geod.polygon_area_perimeter(lon, lat)

    return abs(area) / 1_000_000  # area can be negative depending on the order of vertices
                                  # Convert from square meters to square kilometers

#-------------------------------------------------------------------------------
def am(input):
# validate input structure
    if not valid_am_input(input):
        print("input is not valid")
        return False

    config = input["config"]
    x_length = config["x_length"]
    y_length = config["y_length"]
    square_side = config["square_side"]
#    num_attack_vectors_per_protected_area = config["num_attack_vectors_per_protected_area"]
    target_types = config["target_types"]
    binary_classifications = config["binary_classifications"]
    area_of_interest = input["area_of_interest"]
    noSqAreaOfInterest = len(area_of_interest)
    coverage_areas = input["coverage_areas"]
    sensor_types = input["sensor_types"]
    sensors = input["sensors"]
    noSensors = len(sensors.keys())
    protected_areas = input["protected_areas"]
    flag_coverage_per_square = input["flag_coverage_per_square"]
    #eRadius = config["earth_radius"]
#----------------------------------------
# Calculate the lgPnd_TH for each binary classification & update the config dictionary
    for bc in binary_classifications:
        Pd_TH = binary_classifications[bc]["Pd_TH"]
        lgPnd_TH = math.log10(1 - Pd_TH)
        binary_classifications[bc].update({"lgPnd_TH": lgPnd_TH})

#-----------------------------------------
# Generate lon,lat corner vertices for each square in the area_of_interest
    area_of_interest_vertices = generate_lon_lat_vertices(area_of_interest, config)

    #print(area_of_interest_vertices)
    #f = open("areaOfInterest_vertices.json","w")
    #f.write(json.dumps(area_of_interest_vertices))
#------------------------------------------
# Precompute the surface area for each square in the area_of_interest
    area_of_interest_sqAreas = []

    for sq in area_of_interest_vertices:
        square_vertices = list()
        square_vertices.append(tuple([sq["SW"]["long"],sq["SW"]["lat"]]))
        square_vertices.append(tuple([sq["NW"]["long"],sq["NW"]["lat"]]))
        square_vertices.append(tuple([sq["NE"]["long"],sq["NE"]["lat"]]))
        square_vertices.append(tuple([sq["SE"]["long"],sq["SE"]["lat"]]))
        square_vertices.append(tuple([sq["SW"]["long"],sq["SW"]["lat"]]))
        square_area = calculate_polygon_area(square_vertices)
        area_of_interest_sqAreas.append(square_area)

    #print(area_of_interest_sqAreas)
    #f = open("areaOfInterest_sqAreas.json","w")
    #f.write(json.dumps(area_of_interest_sqAreas))
#-------------------------------------------------------------------------------
# update sensors possible locs with lgPnd and coverage flags based on Pd >= Pd_TH

    for s in sensors:
        for pl in sensors[s]["possible_locs"]:
            for target in target_types:
                for bc in binary_classifications:
                    lgPnd_vector = [ 0 for sq in range(noSqAreaOfInterest) ]
                    coverage_flags = [ 0 for sq in range(noSqAreaOfInterest) ]
                    for sq in range(noSqAreaOfInterest):
                        lgPnd = math.log10(1 - pl["coverage_metrics"][target][bc]["Pd"][sq])
                        lgPnd_vector[sq] = lgPnd
                        if lgPnd <= binary_classifications[bc]["lgPnd_TH"]:
                            coverage_flags[sq] = 1
                    pl["coverage_metrics"][target][bc].update({"lgPnd": lgPnd_vector})
                    pl["coverage_metrics"][target][bc].update({"coverage_flags": coverage_flags})

#-------------------------------------------------------------------------------
# initialize per_square_metrics
    def create_metrics_init(noSq):
        return {
            "covered_by": [0 for sq in range(noSq)],
            "lgPnd": [0 for sq in range(noSq)],
            "Pd": [0 for sq in range(noSq)],
            "prob_false_alarm": ["tbd" for sq in range(noSq)],
            "accuracy_of_location": ["tbd" for sq in range(noSq)]
        }

    per_square_metrics = {}

    for target in target_types:
        per_square_metrics[target] = {
            "for_all_sensors": {
                bc: create_metrics_init(noSqAreaOfInterest)
                for bc in binary_classifications
            },
            "by_sensor_type": {
                st: {
                    bc: create_metrics_init(noSqAreaOfInterest)
                    for bc in binary_classifications
                }
                for st in sensor_types
            },
            "by_sensor": {
                s: {
                    bc: create_metrics_init(noSqAreaOfInterest)
                    for bc in binary_classifications
                }
                for s in sensors
            }
        }

#----------------------------------------
# compute: per_square_metrics
# compute per_square_metrics["by_sensor"]

    for target in target_types:
        for s in sensors:
            for bc in binary_classifications:
                for sq in range(noSqAreaOfInterest):
                    covered_by = sum([
                        ( sensors[s]["actual_locs"][pl] *
                          sensors[s]["possible_locs"][pl]["coverage_metrics"][target][bc]["coverage_flags"][sq]
                        )
                        for pl in range(len(sensors[s]["possible_locs"]))
                    ])

                    lgPnd = sum([
                        sensors[s]["actual_locs"][pl] * math.log10(1 - sensors[s]["possible_locs"][pl]["coverage_metrics"][target][bc]["Pd"][sq])
        # @             sensors[s]["actual_locs"][pl] * sensors[s]["possible_locs"][pl]["coverage_metrics"][target][bc]["lgPnd"][sq]
                        for pl in range(len(sensors[s]["possible_locs"]))
                    ])
                    Pd = 1 - pow(10,lgPnd)
        #            if lgPnd <= lgPnd_TH:
        #                flag_coverage = 1
        #            else: flag_coverage = 0

                    per_square_metrics[target]["by_sensor"][s][bc]["covered_by"][sq] = covered_by
                    per_square_metrics[target]["by_sensor"][s][bc]["lgPnd"][sq] = lgPnd
                    per_square_metrics[target]["by_sensor"][s][bc]["Pd"][sq] = Pd
        #           per_square_metrics["by_sensor"][s]["flag_coverage"][sq] = flag_coverage

# add constraint that prob_detection <= 1-epsilon, so that lgPnd >= epsilon and lg is defined

#--------------------------------------------
# compute per_square_metrics["by all sensors"]

    for target in target_types:
        for bc in binary_classifications:
            for sq in range(noSqAreaOfInterest):
                covered_by = sum([
                    per_square_metrics[target]["by_sensor"][s][bc]["covered_by"][sq]
                    for s in sensors
                ])

                lgPnd = sum([
                    per_square_metrics[target]["by_sensor"][s][bc]["lgPnd"][sq]
                    for s in sensors
                ])

                Pd = 1 - pow(10,lgPnd)
        #        if Pd >= Pd_TH:
        #            flag_coverage = 1
        #        else: flag_coverage = 0

                per_square_metrics[target]["for_all_sensors"][bc]["covered_by"][sq] = covered_by
                per_square_metrics[target]["for_all_sensors"][bc]["lgPnd"][sq] = lgPnd
                per_square_metrics[target]["for_all_sensors"][bc]["Pd"][sq] = Pd
        #       per_square_metrics["for_all_sensors"]["flag_coverage"][sq] = flag_coverage

#--------------------------------------------
# compute per_square_metrics["by_sensor_type"]

    for target in target_types:
        for st in sensor_types:
            for bc in binary_classifications:
                for sq in range(noSqAreaOfInterest):
                    covered_by = sum([
                        per_square_metrics[target]["by_sensor"][s][bc]["covered_by"][sq]
                        for s in sensors
                        if sensors[s]["type"] == st
                    ])

                    lgPnd = sum([
                        per_square_metrics[target]["by_sensor"][s][bc]["lgPnd"][sq]
                        for s in sensors
                        if sensors[s]["type"] == st
                    ])

                    Pd = 1 - pow(10,lgPnd)
        #            if Pd >= Pd_TH:
        #                flag_coverage = 1
        #            else:
        #                flag_coverage = 0

                    per_square_metrics[target]["by_sensor_type"][st][bc]["covered_by"][sq] = covered_by
                    per_square_metrics[target]["by_sensor_type"][st][bc]["lgPnd"][sq] = lgPnd
                    per_square_metrics[target]["by_sensor_type"][st][bc]["Pd"][sq] = Pd
        #           per_square_metrics["by_sensor_type"][st]["flag_coverage"][sq] = flag_coverage

#-------------------------------------------------------------------------------
# coverage metrics help functions:

    def area_coverage(area, target):
#        dgal.debug("area = ", area)
#        dgal.debug("area_size = ", len(area))
#        dgal.debug("flag_coverage_per_square", flag_coverage_per_square)
        noSqInArea = len(area)

        area_coverage = {
            "for_all_sensors": {
                bc: {
                    "coverage_percentage": sum([ flag_coverage_per_square[target]["for_all_sensors"][bc][sq] for sq in area
                        ]) / noSqInArea,
                    "avg_redundancy": sum([ per_square_metrics[target]["for_all_sensors"][bc]["covered_by"][sq] for sq in area
                        ]) / noSqInArea
                }
                for bc in binary_classifications
            },
            "by_sensor_type":  {
                st : {
                    bc: {
                        "coverage_percentage": sum([ flag_coverage_per_square[target]["by_sensor_type"][st][bc][sq] for sq in area
                            ]) / noSqInArea,
                        "avg_redundancy": sum([ per_square_metrics[target]["by_sensor_type"][st][bc]["covered_by"][sq] for sq in area
                            ]) / noSqInArea
                    }
                    for bc in binary_classifications
                }
                for st in sensor_types
            },
            "by_sensor":  {
                s : {
                    bc: {
                        "coverage_percentage": sum([ flag_coverage_per_square[target]["by_sensor"][s][bc][sq] for sq in area
                            ]) / noSqInArea,
                        "avg_redundancy": sum([ per_square_metrics[target]["by_sensor"][s][bc]["covered_by"][sq] for sq in area
                            ]) / noSqInArea
                    }
                    for bc in binary_classifications
                }
                for s in sensors
            }
        }

        return area_coverage
    # use it for both area_of_interest and protected areas; it is basically the same computation
#-------------------------------------------

# compute coverage_areas_metrics

    coverage_areas_metrics = {
        "area_of_interest": {
            target:  area_coverage([sq for sq in range(noSqAreaOfInterest)], target)
            for target in target_types
        },
        "by_coverage_area": {
            ca: {
                target: area_coverage(coverage_areas[ca]["area"], target)
                for target in target_types
            }
            for ca in coverage_areas
        }
    }

#-------------------------------------------------------------------------------
# compute sensor metrics
    metrics_per_sensor = {
        s: {
            "no_units": 0,    # initialized; to be computed next
            "cost": 0,        # initialized; to be computed next
            "ppu": sensors[s]["ppu"],
            "detection_period": sensors[s]["detection_period"],
            "type": sensors[s]["type"],
            "make": sensors[s]["make"],
            "model": sensors[s]["model"],
            "possible_locs": [
                { "long": pl["long"], "lat": pl["lat"]}
                for pl in sensors[s]["possible_locs"]
            ],
            "actual_locs": sensors[s]["actual_locs"]
        }
        for s in sensors
    }
    for s in sensors:
        metrics_per_sensor[s]["no_units"] = \
            sum([ f for f in sensors[s]["actual_locs"] ])
        metrics_per_sensor[s]["cost"] = metrics_per_sensor[s]["ppu"] \
            * metrics_per_sensor[s]["no_units"]

# compute all sensor metrics
    sensor_metrics = {
        "total": {
            "no_units": sum([
                metrics_per_sensor[s]["no_units"]
                for s in sensors
            ]),
            "cost": sum([
                metrics_per_sensor[s]["cost"]
                for s in sensors
            ])
        },
        "per_type": {
            st: {
                "no_units": sum([
                    metrics_per_sensor[s]["no_units"]
                    for s in sensors
                    if sensors[s]["type"] == st
                ]),
                "cost": sum([
                    metrics_per_sensor[s]["cost"]
                    for s in sensors
                    if sensors[s]["type"] == st
                ])
            }
            for st in sensor_types
        },
        "per_sensor": metrics_per_sensor
    }

#-------------------------------------------------------------------------------
# ptrm as a shortcut
    ptrm = protected_areas

# update ptrm with reaction_time_area and detection_time_area

    for a in ptrm["by_protected_area"]:
        area = ptrm["by_protected_area"][a]

        for target in target_types:
            reaction_area = []
            detection_area = []

            reaction_time = area["per_target_type"][target]["reaction_time_LB"]
            reaction_dist = reaction_time * target_types[target]["max_velocity"]
            detection_time = area["per_target_type"][target]["detection_time_allocation"]
            detection_dist = detection_time * target_types[target]["max_velocity"]

            for angleInd in range(ptrm["num_radial_attack_angles"]):
                protected_dist = area["protected_area"][angleInd]
                reaction_area_dist = protected_dist + reaction_dist
                detection_area_dist = reaction_area_dist + detection_dist
                reaction_area.append(reaction_area_dist)
                detection_area.append(detection_area_dist)

            area["per_target_type"][target].update({
                "reaction_area": reaction_area,
                "detection_area": detection_area
            })
            #area
            #dgal.debug("reaction_area", reaction_area)
            #dgal.debug("area", area)

#-------------------------------------------------------------------------------
# update ptrm with vertices of all areas

    def area2longLatVertices(center, area):
        vertices = []
        num_angles = ptrm["num_radial_attack_angles"]
        delta = 360 / num_angles
        for angleInd in range(num_angles):
            angle = delta * angleInd
            dist = area[angleInd]
            longLatPair = longLat(center["long"], center["lat"], angle, dist)
            vertices.append(longLatPair)
        return vertices

    for a in ptrm["by_protected_area"]:
        area = ptrm["by_protected_area"][a]
        protected_area_vertices = area2longLatVertices(area["center"], area["protected_area"])
        area.update({ "protected_area_vertices": protected_area_vertices})

        for target in target_types:
            reaction_area_vertices = area2longLatVertices(area["center"], area["per_target_type"][target]["reaction_area"])
            detection_area_vertices = area2longLatVertices(area["center"], area["per_target_type"][target]["detection_area"])

            area["per_target_type"][target].update({
                "reaction_area_vertices": reaction_area_vertices,
                "detection_area_vertices": detection_area_vertices
            })

# ------------------------------------------------------------------------------
# Functions to compute the coverage for reaction_area & detection_area

    def is_vertex_in_polygon(vertex, polygon_vertices, source_crs='epsg:4326', target_crs='epsg:3395'):
        """
        Check if a vertex is inside a polygon, considering the Earth's curvature by
        transforming coordinates to a projected system before the check.

        Parameters:
        - vertex: The [longitude, latitude] of the point to check.
        - polygon_vertices: A list of [longitude, latitude] coordinates defining the polygon.
        - source_crs (str): The EPSG code of the original geographic coordinate system.
        - target_crs (str): The EPSG code of the target planar coordinate system.
        """
        # Create transformers for coordinate conversions
        transformer_to_projected = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        transformer_to_geographic = Transformer.from_crs(target_crs, source_crs, always_xy=True)

        # Transform polygon vertices to the target CRS
        projected_polygon = transform(transformer_to_projected.transform, Polygon(polygon_vertices))

        # Transform the vertex to the target CRS
        projected_vertex = transform(transformer_to_projected.transform, Point(vertex))

        # Perform the point-in-polygon check on the transformed geometries
        is_inside = projected_polygon.contains(projected_vertex)
        return is_inside

    # Calculate the geographic midpoint between two points using pyproj
    def calculate_geo_midpoint(point1, point2):
        geod = Geod(ellps="WGS84")
        # Get the forward azimuth, back azimuth, and distance from point1 to point2
        fwd_azimuth, back_azimuth, distance = geod.inv(point1[0], point1[1], point2[0], point2[1])
        # Use the fwd function to find the midpoint on the geodesic
        lon, lat, _ = geod.fwd(point1[0], point1[1], fwd_azimuth, distance / 2)
        return lon, lat

    # Calculate the geo center of a square given its four vertices.
    def square_geo_center(vertices):
        sw = vertices['SW']['long'], vertices['SW']['lat']
        ne = vertices['NE']['long'], vertices['NE']['lat']
        nw = vertices['NW']['long'], vertices['NW']['lat']
        se = vertices['SE']['long'], vertices['SE']['lat']

        # Calculate midpoints of opposite sides
        midpoint_sw_ne = calculate_geo_midpoint(sw, ne)
        midpoint_nw_se = calculate_geo_midpoint(nw, se)

        # Calculate the midpoint between the two midpoints to approximate the square's center
        center = calculate_geo_midpoint(midpoint_sw_ne, midpoint_nw_se)
        return center

    # Calculate the geo center of a square by averaging the coordinates of opposite corners.
    def square_average_center(vertices):
        sw = vertices['SW']
        ne = vertices['NE']

        # Averaging longitude and latitude of opposite corners
        center_long = (sw['long'] + ne['long']) / 2
        center_lat = (sw['lat'] + ne['lat']) / 2
        return center_long, center_lat

    # Compute the coverage metrics for both the reaction_area & detection_area for each protected area & target
    for a in ptrm["by_protected_area"]:
        area = ptrm["by_protected_area"][a]

        for target in target_types:
            reaction_area_vertices = area["per_target_type"][target]["reaction_area_vertices"]
            detection_area_vertices = area["per_target_type"][target]["detection_area_vertices"]
            reaction_area_squares = []
            detection_area_squares = []

            for sq_index, sq_vertices in enumerate(area_of_interest_vertices):
                center_point = square_average_center(sq_vertices)
                if is_vertex_in_polygon(center_point, reaction_area_vertices):
                    reaction_area_squares.append(sq_index)
                if is_vertex_in_polygon(center_point, detection_area_vertices):
                    detection_area_squares.append(sq_index)
            #print(reaction_area_squares)
            #print(detection_area_squares)

            reaction_area_coverage = area_coverage(reaction_area_squares, target)
            detection_area_coverage = area_coverage(detection_area_squares, target)

            area["per_target_type"][target].update({
                "reaction_area_coverage": reaction_area_coverage,
                "detection_area_coverage": detection_area_coverage
            })

# ------------------------------------------------------------------------------
# precompute forward and backward index structures
# compute all x indecies from area of interest;
#    xIndList = [ coord[0] for coord in area_of_interest]
# init sqInd_per_coords
#    sqInd_per_coords = { xInd : {} for xInd in xIndList}
# compute sqInd_per_coords
#    for ind in range(len(area_of_interest)):
#        sqInd_per_coords[area_of_interest[ind][0]].update({area_of_interest[ind][1]: ind})
#
#    def sqIndex(xInd,yInd):
#        try:
#            sqInd = sqInd_per_coords[xInd][yInd]
#        except:
#            print("INPUT ERROR: area_of_interest does not contain protected areas")
#            return(-1)
#        else:
#            return(sqInd)
#    print(sqInd_per_coords)
# ----------------------------------------------
# Replaced the above code with a simplified version
# Initialize a dictionary to map (x, y) to their index in area_of_interest
    coord_to_index = {(coord[0], coord[1]): index for index, coord in enumerate(area_of_interest)}

    def sqIndex(xInd, yInd):
        try:
            # Retrieve the index using the (xInd, yInd) tuple directly
            sqInd = coord_to_index[(xInd, yInd)]
        except KeyError:
            # Handle case where the (x, y) is not in area_of_interest
            print("INPUT ERROR: area_of_interest does not contain the specified (x, y) location")
            return -1
        else:
            return sqInd

#-------------------------------------------------------------------------------
# compute lgPnd_area_attack per areaId, target, sensor, binClass and angleInd:

    def lgPnd_area_attack(areaId, target, sensor, binClass, angleInd):
        #area = protected_areas["by_protected_area"][areaId]
        area = ptrm["by_protected_area"][areaId]
        #reaction_time = area["reaction_time_LB"]
        #reaction_dist = reaction_time * config["max_UAS_velocity"]
        #protected_dist = area["protected_area"][angleInd]
        #reaction_area_dist = protected_dist + reaction_dist
        reaction_area_dist = area["per_target_type"][target]["reaction_area"][angleInd]
        detection_area_dist = area["per_target_type"][target]["detection_area"][angleInd]
        if reaction_area_dist >= detection_area_dist:
            return 0
# in this case Pd = 0, Pnd = 1, and lgPnd = 0 to be returned
        detection_distance = detection_area_dist - reaction_area_dist
        avail_detection_time = detection_distance / target_types[target]["max_velocity"]
        sensor_detection_period = sensors[sensor]["detection_period"]
        num_detections = math.floor(avail_detection_time / sensor_detection_period)

        #num_angles = protected_areas["num_radial_attack_angles"]
        num_angles = ptrm["num_radial_attack_angles"]
        segment_length = sensor_detection_period * target_types[target]["max_velocity"]
        delta = 360 / num_angles
        angle = delta * angleInd
# center is given in long/lat degrees
        long_center = area["center"]["long"]
        lat_center = area["center"]["lat"]

# find long & lat of segment i starting point
        def longLat_seg(i):
            dist = reaction_area_dist + i * segment_length
            lonLat = longLat(long_center, lat_center, angle, dist)
            return lonLat
# find lgPnd of the square in which longLat = [long, lat] point is located
        def square_sensor_lgPnd(longLat):
            long0 = config["south_west_corner"]["long"]
            lat0 = config["south_west_corner"]["lat"]
            xInd = math.floor((longLat[0] - long0) / square_side)
            yInd = math.floor((longLat[1] - lat0) / square_side)
            sqInd = sqIndex(xInd,yInd)
            #sqInd = sqInd_per_coords[xInd][yInd]
            if sqInd == -1:
                return 0
            return per_square_metrics[target]["by_sensor"][sensor][binClass]["lgPnd"][sqInd]

        lgPnd_per_segment = []
        for i in range(num_detections):
            longLatBeg = longLat_seg(i)
            longLatEnd = longLat_seg(i+1)
            longMid = (longLatBeg[0] + longLatEnd[0]) / 2
            latMid = (longLatBeg[1] + longLatEnd[1]) / 2
            lgPnd_per_segment.append(square_sensor_lgPnd([longMid, latMid]))

        aggr_lgPnd = sum(lgPnd_per_segment)
        # dgal.debug("areaId", areaId)
        # dgal.debug("sensor", sensor)
        # dgal.debug("angleInd", angleInd)
        # dgal.debug("aggr_lgPnd from lgPnd_area_attack", aggr_lgPnd)
        return(aggr_lgPnd)

#-------------------------------------------------------------------------------
# update ptrm with computed lgPnd_per_attack_angle per area, sensor

    for a in ptrm["by_protected_area"]:
        for t in target_types:
            for s in sensors:
                for bc in binary_classifications:
                    lgPnd_vector = [
                        lgPnd_area_attack(a, t, s, bc, angleInd)
                        for angleInd in range(ptrm["num_radial_attack_angles"])
                    ]
                    ptrm["by_protected_area"][a]["per_target_type"][t]["by_sensor"][s][bc].update(
                        {"lgPnd_per_attack_angle": lgPnd_vector}
                    )
# and corresponding bound constraints
    dgal.debug("ptrm",ptrm)
    lgPnd_bounds_per_area_sensor = dgal.all([
        ptrm["by_protected_area"][a]["per_target_type"][t]["by_sensor"][s][bc]["lgPnd_UB"] >=
        ptrm["by_protected_area"][a]["per_target_type"][t]["by_sensor"][s][bc]["lgPnd_per_attack_angle"][angleInd]
        for a in ptrm["by_protected_area"]
        for t in target_types
        for s in sensors
        for bc in binary_classifications
        for angleInd in range(ptrm["num_radial_attack_angles"])
    ])

# -----------------------------------
# update ptrm with computed ldPnd_per_attack_angle per area, sensor_type

    for a in ptrm["by_protected_area"]:
        for t in target_types:
            for st in sensor_types:
                for bc in binary_classifications:
                    lgPnd_vector = [
                        sum([ ptrm["by_protected_area"][a]["per_target_type"][t]["by_sensor"][s][bc]["lgPnd_per_attack_angle"][angleInd]
                            for s in sensors if sensors[s]["type"] == st
                        ])
                        for angleInd in range(ptrm["num_radial_attack_angles"])
                    ]
                    ptrm["by_protected_area"][a]["per_target_type"][t]["by_sensor_type"][st][bc].update(
                        {"lgPnd_per_attack_angle": lgPnd_vector}
                    )
# and corresponding bound constraints
    lgPnd_bounds_per_area_sensor_type = dgal.all([
        ptrm["by_protected_area"][a]["per_target_type"][t]["by_sensor_type"][st][bc]["lgPnd_UB"] >=
        ptrm["by_protected_area"][a]["per_target_type"][t]["by_sensor_type"][st][bc]["lgPnd_per_attack_angle"][angleInd]
        for a in ptrm["by_protected_area"]
        for t in target_types
        for st in sensor_types
        for bc in binary_classifications
        for angleInd in range(ptrm["num_radial_attack_angles"])
    ])

#-------------------------------------
# update ptrm with computed lgPnd_per_attack_angle per area for all sensors

    for a in ptrm["by_protected_area"]:
        for t in target_types:
            for bc in binary_classifications:
                lgPnd_vector = [
                    sum([ ptrm["by_protected_area"][a]["per_target_type"][t]["by_sensor"][s][bc]["lgPnd_per_attack_angle"][angleInd]
                        for s in sensors
                    ])
                    for angleInd in range(ptrm["num_radial_attack_angles"])
                ]
                ptrm["by_protected_area"][a]["per_target_type"][t]["for_all_sensors"][bc].update(
                    {"lgPnd_per_attack_angle": lgPnd_vector}
                )
# and the corresponding bound constraints
    lgPnd_bounds_per_area_all_sensors = dgal.all([
        ptrm["by_protected_area"][a]["per_target_type"][t]["for_all_sensors"][bc]["lgPnd_UB"] >=
        ptrm["by_protected_area"][a]["per_target_type"][t]["for_all_sensors"][bc]["lgPnd_per_attack_angle"][angleInd]
        for a in ptrm["by_protected_area"]
        for t in target_types
        for bc in binary_classifications
        for angleInd in range(ptrm["num_radial_attack_angles"])
    ])

#-------------------------------------------------------------------------------
# re-arrange the order of keys in area dictionaries

    for a in ptrm["by_protected_area"]:
        area = ptrm["by_protected_area"][a]
        reordered1 = {
            "center": area["center"],
            "protected_area": area["protected_area"],
            "protected_area_vertices": area["protected_area_vertices"],
            "per_target_type": area["per_target_type"]
        }
        ptrm["by_protected_area"][a] = reordered1
        for t in target_types:
            target = area["per_target_type"][t]
            reordered2 = {
                "reaction_time_LB": target["reaction_time_LB"],
                "detection_time_allocation": target["detection_time_allocation"],
                "reaction_area": target["reaction_area"],
                "detection_area": target["detection_area"],
                "reaction_area_vertices": target["reaction_area_vertices"],
                "detection_area_vertices": target["detection_area_vertices"],
                "reaction_area_coverage": target["reaction_area_coverage"],
                "detection_area_coverage": target["detection_area_coverage"],
                "for_all_sensors": target["for_all_sensors"],
                "by_sensor_type": target["by_sensor_type"],
                "by_sensor": target["by_sensor"]
            }
            ptrm["by_protected_area"][a]["per_target_type"][t] = reordered2
    #f = open("chk.json","w")
    #f.write(json.dumps(ptrm["by_protected_area"]))
#-------------------------------------------------------------------------------
# express relevant constraints
    binary_bounds = dgal.all([
        dgal.all([
            dgal.all([ 0 <= f, f <= 1])
            for s in sensors
            for f in sensors[s]["actual_locs"]
        ]),
        dgal.all([
            dgal.all([ 0 <= f, f <= 1])
            for t in target_types
            for bc in binary_classifications
            for f in flag_coverage_per_square[t]["for_all_sensors"][bc]
        ]),
        dgal.all([
            dgal.all([ 0 <= f, f <= 1 ])
            for t in target_types
            for st in sensor_types
            for bc in binary_classifications
            for f in flag_coverage_per_square[t]["by_sensor_type"][st][bc]
        ]),
        dgal.all([
            dgal.all([ 0 <= f, f <= 1 ])
            for t in target_types
            for s in sensors
            for bc in binary_classifications
            for f in flag_coverage_per_square[t]["by_sensor"][s][bc]
        ])
    ])
    dgal.debug("binary_bounds",binary_bounds)

#---------------------------------
    def flagIffXgeZero(flag,x,lb,ub,eps):
    # assumptions: binary flag, X within two bounds, X<0 --> (x < -eps), for (small) eps>0
        flagImplyGeZero = ( x >= lb * (1 - flag) )
        oppositeImplication = (x + eps <= ub * flag)
        return dgal.all([flagImplyGeZero, oppositeImplication])

#>>>>>> I am here: the flag_iff ... function is buggy
#    def flag_iff_lgPnd_LE_UB(f, lgPnd, lgPnd_TH):
#        a = (f == 1)
#        b = (lgPnd <= lgPnd_TH)
#        return iff(a,b)

    def flag_iff_lgPnd_LE_UB(f, lgPnd, lgPnd_TH):
        eps = 0.00001
        lb = -10000.0
        ub = 10000.0
        return flagIffXgeZero(f, (lgPnd_TH - lgPnd), lb, ub, eps)

#---------------------------------
# per square flags match prob_detection_threashold
# re-write it in regular for-loop style and dgal.all aggregate
    per_sq_flags_all_sensors = True
    for t in target_types:
        for bc in binary_classifications:
            lgPnd_TH = binary_classifications[bc]["lgPnd_TH"]
            dgal.debug("lgPnd_TH", lgPnd_TH)

            for sq in range(noSqAreaOfInterest):
                lgPnd_all_sensors = per_square_metrics[t]["for_all_sensors"][bc]["lgPnd"][sq]
                f_all_sensors = flag_coverage_per_square[t]["for_all_sensors"][bc][sq]
                flag_constraint = flag_iff_lgPnd_LE_UB(f_all_sensors, lgPnd_all_sensors, lgPnd_TH)
                per_sq_flags_all_sensors = dgal.all([per_sq_flags_all_sensors, flag_constraint])
        # the rest in the loop for debug
                if not(flag_constraint):
                    dgal.debug("target",t)
                    dgal.debug("binClass",bc)
                    dgal.debug("sq",sq)
                    dgal.debug("lgPnd_all_sensors",lgPnd_all_sensors)
                    dgal.debug("f_all_sensors",f_all_sensors)
        #            dgal.debug("computed_flag",per_square_metrics["for_all_sensors"]["flag_coverage"][sq])
                    dgal.debug("flag_constraint", flag_constraint)

    per_square_flags_match_prob_det_threashold = dgal.all([
        dgal.all([
            flag_iff_lgPnd_LE_UB(
                                flag_coverage_per_square[t]["for_all_sensors"][bc][sq],
                                per_square_metrics[t]["for_all_sensors"][bc]["lgPnd"][sq],
                                binary_classifications[bc]["lgPnd_TH"])
            for t in target_types
            for bc in binary_classifications
            for sq in range(noSqAreaOfInterest)
        ]),
        dgal.all([
            flag_iff_lgPnd_LE_UB(
                                flag_coverage_per_square[t]["by_sensor_type"][st][bc][sq],
                                per_square_metrics[t]["by_sensor_type"][st][bc]["lgPnd"][sq],
                                binary_classifications[bc]["lgPnd_TH"])
            for t in target_types
            for st in sensor_types
            for bc in binary_classifications
            for sq in range(noSqAreaOfInterest)
        ]),
        dgal.all([
            flag_iff_lgPnd_LE_UB(
                                flag_coverage_per_square[t]["by_sensor"][s][bc][sq],
                                per_square_metrics[t]["by_sensor"][s][bc]["lgPnd"][sq],
                                binary_classifications[bc]["lgPnd_TH"])
            for t in target_types
            for s in sensors
            for bc in binary_classifications
            for sq in range(noSqAreaOfInterest)
        ])
    ])
    dgal.debug("per_square_flags_match_prob_det_threashold",per_square_flags_match_prob_det_threashold)

#---------------------------------
# constraints on lgPnd bounds for protected areas:
# lgPnd bounds for all protected areas for all sensors

    lgPnd_bounds_for_all_protected_areas_and_all_sensors = dgal.all([
        ( ptrm["by_protected_area"][a]["per_target_type"][t]["for_all_sensors"][bc]["lgPnd_UB"] <=
          ptrm["all_protected_areas"]["per_target_type"][t]["for_all_sensors"][bc]["lgPnd_UB"]
        )
        for a in ptrm["by_protected_area"]
        for t in target_types
        for bc in binary_classifications
    ])
    dgal.debug("lgPnd_bounds_for_all_protected_areas_and_all_sensors",lgPnd_bounds_for_all_protected_areas_and_all_sensors)

#-------------------------------------------------------------------------------------------
# lgPnd bounds for all protected areas by sensor type

    lgPnd_bounds_for_all_protected_areas_per_sensor_type = dgal.all([
        ( ptrm["by_protected_area"][a]["per_target_type"][t]["by_sensor_type"][st][bc]["lgPnd_UB"] <=
          ptrm["all_protected_areas"]["per_target_type"][t]["by_sensor_type"][st][bc]["lgPnd_UB"]
        )
        for a in ptrm["by_protected_area"]
        for t in target_types
        for st in sensor_types
        for bc in binary_classifications
    ])
    dgal.debug("lgPnd_bounds_for_all_protected_areas_per_sensor_type",lgPnd_bounds_for_all_protected_areas_per_sensor_type)

#--------------
# lgPnd bounds for all protected areas by sensor

    lgPnd_bounds_for_all_protected_areas_per_sensor = dgal.all([
        ( ptrm["by_protected_area"][a]["per_target_type"][t]["by_sensor"][s][bc]["lgPnd_UB"] <=
          ptrm["all_protected_areas"]["per_target_type"][t]["by_sensor"][s][bc]["lgPnd_UB"]
        )
        for a in ptrm["by_protected_area"]
        for t in target_types
        for s in sensors
        for bc in binary_classifications
    ])
    dgal.debug("lgPnd_bounds_for_all_protected_areas_per_sensor",lgPnd_bounds_for_all_protected_areas_per_sensor)

#---------------------------------
# reaction time bounds for all protected areas

    reaction_time_bounds = dgal.all([
        ( ptrm["by_protected_area"][a]["per_target_type"][t]["reaction_time_LB"] >=
          ptrm["all_protected_areas"]["per_target_type"][t]["reaction_time_LB"]
        )
        for a in ptrm["by_protected_area"]
        for t in target_types
    ])
    dgal.debug("reaction_time_bounds",reaction_time_bounds)

#---------------------------------
# aggregate all constraints in the model

    constraints_wout_flags = dgal.all([
        binary_bounds,
        lgPnd_bounds_per_area_sensor,
        lgPnd_bounds_per_area_sensor_type,
        lgPnd_bounds_per_area_all_sensors,
        lgPnd_bounds_for_all_protected_areas_per_sensor,
        lgPnd_bounds_for_all_protected_areas_per_sensor_type,
        lgPnd_bounds_for_all_protected_areas_and_all_sensors,
        reaction_time_bounds
    ])

    constraints = dgal.all([
        constraints_wout_flags,
        per_square_flags_match_prob_det_threashold
    ])
    dgal.debug("constraints",constraints)

    return({
        "title": "sensor assignment metrics",
        "config": config,
        "area_of_interest": area_of_interest,
        "constraints": constraints,
        "constraints_wout_flags": constraints_wout_flags,
        "sensor_metrics": sensor_metrics,
        "per_square_metrics": per_square_metrics,
        "coverage_areas_metrics": coverage_areas_metrics,
        "protected_areas_metrics": ptrm,
        "debug": {
            "specific_constraints": {
                "binary_bounds": binary_bounds,
                "per_square_flags_match_prob_det_threashold": per_square_flags_match_prob_det_threashold,
                "lgPnd_bounds_per_area_sensor": lgPnd_bounds_per_area_sensor,
                "lgPnd_bounds_per_area_sensor_type": lgPnd_bounds_per_area_sensor_type,
                "lgPnd_bounds_per_area_all_sensors": lgPnd_bounds_per_area_all_sensors,
                "lgPnd_bounds_for_all_protected_areas_per_sensor": lgPnd_bounds_for_all_protected_areas_per_sensor,
                "lgPnd_bounds_for_all_protected_areas_per_sensor_type": lgPnd_bounds_for_all_protected_areas_per_sensor_type,
                "lgPnd_bounds_for_all_protected_areas_and_all_sensors": lgPnd_bounds_for_all_protected_areas_and_all_sensors,
                "reaction_time_bounds": reaction_time_bounds,
                "per_sq_flags_all_sensors": per_sq_flags_all_sensors
            },
        }
    })
#-------------------------------------------------------------------------------
# flag for square coverage

def flagPerSq(Pd, Pd_TH):
    if Pd >= Pd_TH:
        return 1
    else:
        return 0

def redundancyAndFusion(coveredBy,flag):
    if flag == 0:
        return -1
    if coveredBy >= 3:
        return 3
    else:
        return coveredBy

#-------------------------------------------------------------------------------

# use this function to extend the model output w/information useful to the user
# and yet would not interfere with optimization

def plannerView(o):
    targetTypes = o["config"]["target_types"]
    binaryClassifications = o["config"]["binary_classifications"]
    noSqs = len(o["area_of_interest"])

# for all_sensors: add flags, and redundancyAndFusion arrays in per_square_metrics
    for tt in o["config"]["target_types"]:
        for bc in o["config"]["binary_classifications"]:
            flags = []
            redundancy_and_fusion = []
            for sq in range(noSqs):
                flag = flagPerSq(
                    o["per_square_metrics"][tt]["for_all_sensors"][bc]["Pd"][sq],
                    binaryClassifications[bc]["Pd_TH"]
                )
                flags.append(flag)
                print("flag:", flag)
                print("covered_by:", o["per_square_metrics"][tt]["for_all_sensors"][bc]["covered_by"][sq])
                redFus = redundancyAndFusion(
                    o["per_square_metrics"][tt]["for_all_sensors"][bc]["covered_by"][sq],
                    flag
                )
                redundancy_and_fusion.append(redFus)
            o["per_square_metrics"][tt]["for_all_sensors"][bc].update({
                "redundancy_and_fusion": redundancy_and_fusion
            })
            o["per_square_metrics"][tt]["for_all_sensors"][bc].update({
                "flags": flags
            })

# by sensor_type: add flags, and redundancyAndFusion arrays in per_square_metrics
    for tt in o["config"]["target_types"]:
        for bc in o["config"]["binary_classifications"]:
            for st in o["sensor_metrics"]["per_type"]:
                flags = []
                redundancy_and_fusion = []
                for sq in range(noSqs):
                    flag = flagPerSq(
                        o["per_square_metrics"][tt]["by_sensor_type"][st][bc]["Pd"][sq],
                        binaryClassifications[bc]["Pd_TH"]
                    )
                    flags.append(flag)
                    redFus = redundancyAndFusion(
                        o["per_square_metrics"][tt]["by_sensor_type"][st][bc]["covered_by"][sq],
                        flag
                    )
                    redundancy_and_fusion.append(redFus)
                o["per_square_metrics"][tt]["by_sensor_type"][st][bc].update({
                    "redundancy_and_fusion": redundancy_and_fusion
                })
                o["per_square_metrics"][tt]["by_sensor_type"][st][bc].update({
                    "flags": flags
                })


# by sensor: add flags, and redundancyAndFusion arrays in per_square_metrics
    for tt in o["config"]["target_types"]:
        for bc in o["config"]["binary_classifications"]:
            for s in o["sensor_metrics"]["per_sensor"]:
                flags = []
                redundancy_and_fusion = []
                for sq in range(noSqs):
                    flag = flagPerSq(
                        o["per_square_metrics"][tt]["by_sensor"][s][bc]["Pd"][sq],
                        binaryClassifications[bc]["Pd_TH"]
                    )
                    flags.append(flag)
                    redFus = redundancyAndFusion(
                        o["per_square_metrics"][tt]["by_sensor"][s][bc]["covered_by"][sq],
                        flag
                    )
                    redundancy_and_fusion.append(redFus)
                o["per_square_metrics"][tt]["by_sensor"][s][bc].update({
                    "redundancy_and_fusion": redundancy_and_fusion
                })
                o["per_square_metrics"][tt]["by_sensor"][s][bc].update({
                    "flags": flags
                })

# add combined_classification view per target type, for all sensors
    combinedClassificationAllSensors = {}
    for tt in o["config"]["target_types"]:
        combinedClassificationAllSensors.update({tt: [] })
        for sq in range(noSqs):
            cov = "none"
            for bc in o["config"]["binary_classifications"]:
                if o["per_square_metrics"][tt]["for_all_sensors"][bc]["flags"][sq] ==  1:
                    cov = bc
            combinedClassificationAllSensors[tt].append(cov)
    o.update({
        "per_square_combined_classifications": combinedClassificationAllSensors
    })



    return o

#-------------------------------------------------------------------------------
# optimize sensor placement for given reqs and objectives
def optiSensors(inputVarCore, optReqs, inputInstance):
    inputVar = inputVarCore
    target_types = inputVar["config"]["target_types"]
    binary_classifications = inputVar["config"]["binary_classifications"]
    noSqAreaOfInterest = len(inputVar["area_of_interest"])
    sensor_types = inputVar["sensor_types"]
    sensors = inputVar["sensors"]
    if len(inputVar["protected_areas"]["by_protected_area"]) == 0:
        no_prot_areas = True
    else:
        no_prot_areas = False

# incorporate inputInstance actual locs as requirements by replacing corrs
# dgalType declarations with 1s

    for s in sensors:
        newActualLocs = []
        for li in range(len(sensors[s]["possible_locs"])):
            if inputInstance["sensors"][s]["actual_locs"][li] == 1:
                newActualLocs.append(1)
            else:
                newActualLocs.append({"dgalType": "int?"})
        sensors[s]["actual_locs"] = newActualLocs

# add flag_coverage_per_square
    for target in target_types:
        flags_all_sensors = {}
        flags_by_sensor_type = {st: {} for st in sensor_types}
        flags_by_sensor = {s: {} for s in sensors}

    for bc in binary_classifications:
        flags_all_sensors[bc] = [{ "dgalType": "int?" } for sq in range(noSqAreaOfInterest)]
        for st in sensor_types:
            flags_by_sensor_type[st][bc] = [{ "dgalType": "int?" } for sq in range(noSqAreaOfInterest)]
        for s in sensors:
            flags_by_sensor[s][bc] = [{ "dgalType": "int?" } for sq in range(noSqAreaOfInterest)]

    flags = {
        "for_all_sensors": flags_all_sensors,
        "by_sensor_type": flags_by_sensor_type,
        "by_sensor": flags_by_sensor
    }

    if "flag_coverage_per_square" not in inputVar:
        inputVar["flag_coverage_per_square"] = {}
        inputVar["flag_coverage_per_square"].update({target: flags})

#------------------------------------------
# optimize

    minMax = optReqs["minMax"]
#------------------------------------------
# These multipliers to make bounds a bit tighter, so that the numerically approximate
# solver solution will satisfy the original constraints in the model
    def tub(ub):
        eps = 0.000001
        if ub > 0:
            return ub * (1 - eps)
        elif ub < 0:
            return ub * (1 + eps)
        else:    # ub == 0
            return -1 * eps

    def tlb(lb):
        eps = 0.000001
        if lb > 0:
            return lb * (1 + eps)
        elif lb < 0:
            return lb * (1 - eps)
        else:  # lb == 0
            return -eps

#------------------------------------------
    def constraints(o):

        if no_prot_areas:
            protected_areas_constraints = True
        else:
            protected_areas_constraints = dgal.all([
                dgal.all([
                    ( o["protected_areas_metrics"]["all_protected_areas"]["per_target_type"][t]["for_all_sensors"][bc]["lgPnd_UB"] <= math.log10(1 - LB) )
                    for t in target_types
                    for bc in binary_classifications
                    for LB in [
                        optReqs["protected_areas_metrics"]["all_protected_areas"]["per_target_type"][t]["for_all_sensors"][bc]["Pd_before_reaction_time"]["LB"]
                    ]
                    if LB != "unbounded"
                ]),
                dgal.all([
                    ( o["protected_areas_metrics"]["by_protected_area"][a]["per_target_type"][t]["for_all_sensors"][bc]["lgPnd_UB"] <= \
                        math.log10(1 - LB)
                    )
                    for a in o["protected_areas_metrics"]["by_protected_area"]
                    for t in target_types
                    for bc in binary_classifications
                    for LB in [
                        optReqs["protected_areas_metrics"]["by_protected_area"][a]["per_target_type"][t]["for_all_sensors"][bc]["Pd_before_reaction_time"]["LB"]
                    ]
                    if LB != "unbounded"
                ])
            ])
# total sensor units constraints

        UB = optReqs["sensor_metrics"]["total"]["no_units"]["UB"]
        if UB != "unbounded":
            total_units_constraint = ( o["sensor_metrics"]["total"]["no_units"] <= UB )
        else:
            total_units_constraint = True

# total cost constraint
        UB = optReqs["sensor_metrics"]["total"]["cost"]["UB"]
        if UB != "unbounded":
            total_cost_constraint = o["sensor_metrics"]["total"]["cost"] <= tub(UB)
        else:
            total_cost_constraint = True

        constraints = dgal.all([
            o["constraints"],
            protected_areas_constraints,
            total_units_constraint,
            total_cost_constraint,
            dgal.all([
                o["sensor_metrics"]["per_sensor"][s]["no_units"] <= UB
                for s in sensors
                for UB in [
                    optReqs["sensor_metrics"]["per_sensor"][s]["no_units"]["UB"]
                ]
                if UB != "unbounded"
            ]),
            dgal.all([
                o["coverage_areas_metrics"]["area_of_interest"][t]["for_all_sensors"][bc]["coverage_percentage"] >= tlb(LB)
                for t in target_types
                for bc in binary_classifications
                for LB in [
                    optReqs["coverage_areas_metrics"]["area_of_interest"][t]["for_all_sensors"][bc]["coverage_percentage"]["LB"]
                ]
                if LB != "unbounded"
            ]),
            dgal.all([
                o["coverage_areas_metrics"]["area_of_interest"][t]["for_all_sensors"][bc]["avg_redundancy"] >= tlb(LB)
                for t in target_types
                for bc in binary_classifications
                for LB in [
                    optReqs["coverage_areas_metrics"]["area_of_interest"][t]["for_all_sensors"][bc]["avg_redundancy"]["LB"]
                ]
                if LB != "unbounded"
            ]),
            dgal.all([
                o["coverage_areas_metrics"]["by_coverage_area"][ca][t]["for_all_sensors"][bc]["coverage_percentage"] >= tlb(LB)
                for ca in o["coverage_areas_metrics"]["by_coverage_area"]
                for t in target_types
                for bc in binary_classifications
                for LB in [
                    optReqs["coverage_areas_metrics"]["by_coverage_area"][ca][t]["for_all_sensors"][bc]["coverage_percentage"]["LB"]
                ]
                if LB != "unbounded"
            ]),
            dgal.all([
                o["coverage_areas_metrics"]["by_coverage_area"][ca][t]["for_all_sensors"][bc]["avg_redundancy"] >= tlb(LB)
                for ca in o["coverage_areas_metrics"]["by_coverage_area"]
                for t in target_types
                for bc in binary_classifications
                for LB in [
                    optReqs["coverage_areas_metrics"]["by_coverage_area"][ca][t]["for_all_sensors"][bc]["avg_redundancy"]["LB"]
                ]
                if LB != "unbounded"
            ])
        ])
        return constraints

#------------------------------------------
    # normalize the objective in the range [0-1]
    def normObjective(objective, objReqs):

        if objReqs["minMax"] == "min":
            normObj = (objReqs["max"] - objective) / (objReqs["max"] - objReqs["min"])
        else:
            normObj = (objective - objReqs["min"]) / (objReqs["max"] - objReqs["min"])

        return normObj
#------------------------------------------
    def obj(o):
        cost = normObjective(o["sensor_metrics"]["total"]["cost"], optReqs["sensor_metrics"]["total"]["cost"]) * \
            optReqs["sensor_metrics"]["total"]["cost"]["obj_weight"]

        units = normObjective(o["sensor_metrics"]["total"]["no_units"], optReqs["sensor_metrics"]["total"]["no_units"]) * \
            optReqs["sensor_metrics"]["total"]["no_units"]["obj_weight"]

        cov_area_of_interest = sum([
            normObjective(o["coverage_areas_metrics"]["area_of_interest"][t]["for_all_sensors"][bc]["coverage_percentage"],
                optReqs["coverage_areas_metrics"]["area_of_interest"][t]["for_all_sensors"][bc]["coverage_percentage"]) * \
                optReqs["coverage_areas_metrics"]["area_of_interest"][t]["for_all_sensors"][bc]["coverage_percentage"]["obj_weight"]
                for t in target_types
                for bc in binary_classifications
        ])
        cov_by_areas = sum([
            normObjective(o["coverage_areas_metrics"]["by_coverage_area"][ca][t]["for_all_sensors"][bc]["coverage_percentage"],
                optReqs["coverage_areas_metrics"]["by_coverage_area"][ca][t]["for_all_sensors"][bc]["coverage_percentage"]) * \
                optReqs["coverage_areas_metrics"]["by_coverage_area"][ca][t]["for_all_sensors"][bc]["coverage_percentage"]["obj_weight"]
                for ca in o["coverage_areas_metrics"]["by_coverage_area"]
                for t in target_types
                for bc in binary_classifications
        ])
        if no_prot_areas:
            lgPnd_reaction_all_areas = 0.0
            lgPnd_reaction_by_areas = 0.0
        else:
            lgPnd_reaction_all_areas = sum([
                normObjective(o["protected_areas_metrics"]["all_protected_areas"]["per_target_type"][t]["for_all_sensors"][bc]["lgPnd_UB"],
                    optReqs["protected_areas_metrics"]["all_protected_areas"]["per_target_type"][t]["for_all_sensors"][bc]["lgPnd_UB"]) * \
                    optReqs["protected_areas_metrics"]["all_protected_areas"]["per_target_type"][t]["for_all_sensors"][bc]["lgPnd_UB"]["obj_weight"]
                for t in target_types
                for bc in binary_classifications
            ])
            lgPnd_reaction_by_areas = sum([
                normObjective(o["protected_areas_metrics"]["by_protected_area"][a]["per_target_type"][t]["for_all_sensors"][bc]["lgPnd_UB"],
                    optReqs["protected_areas_metrics"]["by_protected_area"][a]["per_target_type"][t]["for_all_sensors"][bc]["lgPnd_UB"]) * \
                    optReqs["protected_areas_metrics"]["by_protected_area"][a]["per_target_type"][t]["for_all_sensors"][bc]["lgPnd_UB"]["obj_weight"]
                for a in o["protected_areas_metrics"]["by_protected_area"]
                for t in target_types
                for bc in binary_classifications
            ])

        obj = sum([
            cost,
            units,
            cov_area_of_interest,
            cov_by_areas,
            lgPnd_reaction_all_areas,
            lgPnd_reaction_by_areas
        ])
        return obj

# optimize
    #options = {"problemType": "mip", "solver":"glpk", "debug": True}
    options = {"problemType": "mip", "solver":"gurobi_direct", "debug": True}
    answer = dgal.optimize(am, inputVar, minMax, obj, constraints, options)
    return answer

#-------------------------------------------------------------------------------
# optiSensorMetrics optimizes, and then constructs and returns sensor metrics
# change it to just computing from answers, so that in the main file, you run two functions,
# rather then one.

def sensorMetricsFromOpt(answer):
# note: this function may modify answer, because am may modify input
    if (answer["status"]["dgal_status"] == "ok" and \
        answer["status"]["termination_condition"] == "optimal"):
        output = plannerView(am(answer["solution"]))
        outOpt = {
            "status": answer["status"],
            "sensorPerformanceMetrics": output
        }
    else:
        outOpt = { "status": answer["status"] }
    return outOpt

#------------------------------------------------------------------------------- @ Not used
# optiSensorMetrics optimizes, and then constructs and returns sensor metrics
# change it to just computing from answers, so that in the main file, you run two functions,
# rather then one.

def optiSensorMetrics(inputVarCore, optReqs, inputInstance):
    answer = optiSensors(inputVarCore, optReqs, inputInstance)
    return sensorMetricsFromOpt(answer)


#-------------------------------------------------------------------------------
# sensorMetrics computes a range of coverage metrics given an input with sensor locations
# from the output, use Pd per square (all sensors, per type, per sensor)
# to update flag_coverage_per_square_metrics, and then recompute output from
# updated input.
# assumption: inputInstance must have instantiated actual locs for all possible locs
# in inputVarCore; as well as lgPnd bounds for protected areas

def sensorMetrics(inputVarCore, inputInstance):
    dgal.debug("inputInstance", inputInstance)
    input = inputVarCore
    sensor_types = input["sensor_types"]
    sensors = input["sensors"]
    target_types = input["config"]["target_types"]
    binary_classifications = input["config"]["binary_classifications"]
    noSqAreaOfInterest = len(input["area_of_interest"])

# instantiate actual locs of sensors
    for s in sensors:
        input["sensors"][s]["actual_locs"] = inputInstance["sensors"][s]["actual_locs"]

# instantiate lgPnd_UB for protected areas
    for target in target_types:
        for bc in binary_classifications:
            input["protected_areas"]["all_protected_areas"]["per_target_type"][target]["for_all_sensors"][bc]["lgPnd_UB"] = \
                inputInstance["protected_areas"]["all_protected_areas"]["per_target_type"][target]["for_all_sensors"][bc]["lgPnd_UB"]
            for st in sensor_types:
                input["protected_areas"]["all_protected_areas"]["per_target_type"][target]["by_sensor_type"][st][bc]["lgPnd_UB"] = \
                    inputInstance["protected_areas"]["all_protected_areas"]["per_target_type"][target]["by_sensor_type"][st][bc]["lgPnd_UB"]
            for s in sensors:
                input["protected_areas"]["all_protected_areas"]["per_target_type"][target]["by_sensor"][s][bc]["lgPnd_UB"] = \
                    inputInstance["protected_areas"]["all_protected_areas"]["per_target_type"][target]["by_sensor"][s][bc]["lgPnd_UB"]

    for area in input["protected_areas"]["by_protected_area"]:
        for target in target_types:
            for bc in binary_classifications:
                input["protected_areas"]["by_protected_area"][area]["per_target_type"][target]["for_all_sensors"][bc]["lgPnd_UB"] = \
                    inputInstance["protected_areas"]["by_protected_area"][area]["per_target_type"][target]["for_all_sensors"][bc]["lgPnd_UB"]
                for st in sensor_types:
                    input["protected_areas"]["by_protected_area"][area]["per_target_type"][target]["by_sensor_type"][st][bc]["lgPnd_UB"] = \
                        inputInstance["protected_areas"]["by_protected_area"][area]["per_target_type"][target]["by_sensor_type"][st][bc]["lgPnd_UB"]
                for s in sensors:
                    input["protected_areas"]["by_protected_area"][area]["per_target_type"][target]["by_sensor"][s][bc]["lgPnd_UB"] = \
                        inputInstance["protected_areas"]["by_protected_area"][area]["per_target_type"][target]["by_sensor"][s][bc]["lgPnd_UB"]

# add flag_coverage_per_square
    for target in target_types:
        flags_all_sensors = {}
        flags_by_sensor_type = {st: {} for st in sensor_types}
        flags_by_sensor = {s: {} for s in sensors}

        for bc in binary_classifications:
            flags_all_sensors[bc] = [1 for sq in range(noSqAreaOfInterest)]
            for st in sensor_types:
                flags_by_sensor_type[st][bc] = [1 for sq in range(noSqAreaOfInterest)]
            for s in sensors:
                flags_by_sensor[s][bc] = [1 for sq in range(noSqAreaOfInterest)]

        flags = {
            "for_all_sensors": flags_all_sensors,
            "by_sensor_type": flags_by_sensor_type,
            "by_sensor": flags_by_sensor
            }

        if "flag_coverage_per_square" not in input:
            input["flag_coverage_per_square"] = {}
        input["flag_coverage_per_square"].update({target: flags})

    dgal.debug("input",input)
    output = am(input)
    #f = open("chk.json","w")
    #f.write(json.dumps(output))

# update flag_coverage_per_square_metrics
    for target in target_types:
        flags_all_sensors = {}
        flags_by_sensor_type = {st: {} for st in sensor_types}
        flags_by_sensor = {s: {} for s in sensors}

        for bc in binary_classifications:
            flags_all_sensors[bc] = [
                flagPerSq(output["per_square_metrics"][target]["for_all_sensors"][bc]["Pd"][sq], binary_classifications[bc]["Pd_TH"] )
                for sq in range(noSqAreaOfInterest)
            ]
            for st in sensor_types:
                flags_by_sensor_type[st][bc] = [
                    flagPerSq(output["per_square_metrics"][target]["by_sensor_type"][st][bc]["Pd"][sq], binary_classifications[bc]["Pd_TH"] )
                    for sq in range(noSqAreaOfInterest)
                ]
            for s in sensors:
                flags_by_sensor[s][bc] = [
                    flagPerSq(output["per_square_metrics"][target]["by_sensor"][s][bc]["Pd"][sq], binary_classifications[bc]["Pd_TH"] )
                    for sq in range(noSqAreaOfInterest)
                ]

        flags = {
            "for_all_sensors": flags_all_sensors,
            "by_sensor_type": flags_by_sensor_type,
            "by_sensor": flags_by_sensor
            }
        input["flag_coverage_per_square"].update({target: flags})


    output = plannerView(am(input))

    return output

#-------------------------------------------------------------------------------
