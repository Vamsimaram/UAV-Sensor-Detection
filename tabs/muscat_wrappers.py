#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import math
#sys.path.append("/Users/alexbrodsky/opt/anaconda3")
sys.path.append("/Users/talmanie/Desktop/Sensor_Optimization/sensor_optimization_v5")

current_dir = os.path.dirname(os.path.abspath(__file__))
dgal_path = os.path.join(current_dir, "aaa_lib_dgalPy")
sys.path.insert(0, dgal_path)
import dgalPy as dgal

import sensorAssignmentModel as sa
import json
import copy

dgal.startDebug()
def verifyInputVar(muscatInputVarCore):
    return True
def verifyInputInstance(muscatInputInstance, muscatInputVarCore):
    return True
def verifyOptReqs(muscatOptReqs, muscatInputVarCore):
    return True
maxPd = 0.999999999

# -------------------------------------------------------------------
# muscat input wrap functions
def wrapInputVarCore(muscatInputVarCore):

# init
    inputVarCore = copy.deepcopy(muscatInputVarCore)
    config = inputVarCore["config"]
    square_side = config["square_side"]
    latSWcorner = config["south_west_corner"]["lat"]
    longSWcorner = config["south_west_corner"]["long"]
    areaOfInterest = inputVarCore["area_of_interest"]
    sensors = inputVarCore["sensors"]
    sensorTypes = inputVarCore["sensor_types"]

    print("Processing area of interest coordinates...")
    print(f"SW Corner: lat={latSWcorner}, long={longSWcorner}")
    print(f"Square side: {square_side}")

# change format of area_of_interest to cell [i,j] indices
    updated_aoi = []
    for ind in range(len(areaOfInterest)):
        # Handle both [lat, long] and [long, lat] formats
        if isinstance(areaOfInterest[ind], list) and len(areaOfInterest[ind]) == 2:
            # Check if this looks like [lat, long] format (lat typically 30-50 in US, long typically -120 to -70)
            coord1, coord2 = areaOfInterest[ind]
            if abs(coord1) < abs(coord2) and coord2 < 0:  # likely [lat, long] format
                lat, long = coord1, coord2
                print(f"Detected [lat, long] format for coordinate {ind}")
            else:  # assume [long, lat] format
                long, lat = coord1, coord2
                print(f"Detected [long, lat] format for coordinate {ind}")
        else:
            print(f"Invalid coordinate format at index {ind}: {areaOfInterest[ind]}")
            continue
            
        # Calculate grid indices
        x = (long - longSWcorner) / square_side
        y = (lat - latSWcorner) / square_side
        
        # Round to nearest integer instead of using int() which truncates
        i = round(x)
        j = round(y)
        
        if ind <= 6:
            print(f"Coordinate {ind}:")
            print(f"  lat={lat}, long={long}")
            print(f"  x={x}, y={y}")
            print(f"  i={i}, j={j}")
            print(f"  Error in x: {abs(x - i)}, Error in y: {abs(y - j)}")

        # Check if cells are reasonably aligned (increased tolerance for floating point errors)
        errorTH = 0.5  # Increased from 0.1 to 0.5 to handle floating point precision
        if abs(x - i) > errorTH or abs(y - j) > errorTH:
            print(f"\nWARNING: Cell alignment issue at coordinate {ind}")
            print(f"  Expected grid position: ({i}, {j})")
            print(f"  Actual position: ({x:.6f}, {y:.6f})")
            print(f"  Error: x_error={abs(x-i):.6f}, y_error={abs(y-j):.6f}")
            print("  Continuing with rounded values...")
            
        updated_aoi.append([i, j])

    inputVarCore["area_of_interest"] = updated_aoi
    print(f"Updated area_of_interest: {updated_aoi}")

# replace file refs to Pd arrays with the actual arrays
    for s in sensors:
        for l in sensors[s]["possible_locs"]:
            for tt in l["coverage_metrics"]:
                for bc in l["coverage_metrics"][tt]:
                    bcData = l["coverage_metrics"][tt][bc]
                    if "Pd" in bcData and "@file" in str(bcData["Pd"]):
                        fLink = bcData["Pd"]["@file"]
                        try:
                            with open(fLink, "r") as file:
                                PdArray = json.loads(file.read())
                            bcData["Pd"] = PdArray
                            print(f"Successfully loaded Pd array from {fLink}")
                        except FileNotFoundError:
                            print(f"Warning: Could not find file {fLink}")
                            # Use default values or handle gracefully
                            bcData["Pd"] = [0.5] * len(updated_aoi)  # Default Pd values
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse JSON from {fLink}")
                            bcData["Pd"] = [0.5] * len(updated_aoi)  # Default Pd values

# add actual locs as dgalType declarations
    for s in sensors:
        sensors[s]["actual_locs"] = [
            { "dgalType": "int?" }
            for l in sensors[s]["possible_locs"]
        ]

# add protected area structure, with empty set of protected areas
    inputVarCore.update({
        "protected_areas": {
            "num_radial_attack_angles": 12,
            "all_protected_areas": {
                "per_target_type": {
                    tt: {
                      "reaction_time_LB": 0.0,
                      "for_all_sensors": {
                            bc : {"lgPnd_UB": 0.0}
                            for bc in config["binary_classifications"]
                      },
                      "by_sensor_type": {
                            st: {
                                bc : {"lgPnd_UB": 0.0}
                                for bc in config["binary_classifications"]
                            }
                            for st in sensorTypes
                      },
                      "by_sensor": {
                            s: {
                                bc : {"lgPnd_UB": 0.0}
                                for bc in config["binary_classifications"]
                            }
                            for s in sensors
                      }
                    }
                    for tt in config["target_types"]
                }
            },
            "by_protected_area": {}
        }
    })

# in case there is a Pd = 1, modify it to maxPd = 1 - small epsilon
    for s in sensors:
        for l in sensors[s]["possible_locs"]:
            for tt in l["coverage_metrics"]:
                for bc in l["coverage_metrics"][tt]:
                    if "Pd" in l["coverage_metrics"][tt][bc]:
                        PdArray = l["coverage_metrics"][tt][bc]["Pd"]
                        for i in range(len(PdArray)):
                            if PdArray[i] >= 1.0:
                                PdArray[i] = maxPd
                                print(f"Adjusted Pd value from 1.0 to {maxPd}")

    return inputVarCore

# end of wrapInputVarCore
# --------------------------------------------------------------------

def wrapInputInstance(muscatInputInstance, inputVarCore):
# muscatInputInstance may be modified
    inputInstance = muscatInputInstance
    inputInstance.update({ "protected_areas": inputVarCore["protected_areas"]})
    return inputInstance

#----------------------------------------------------------------------
def wrapOptReqs(muscatOptReqs, inputVarCore):
# muscatOptReqs may be modified
# init optReqs
    mor = muscatOptReqs
    resourceMetrics = mor["resource_metrics"]
    cost = resourceMetrics["cost"]
    perSensor = resourceMetrics["per_sensor"]
    performanceMetrics = mor["performance_metrics"]
    config = inputVarCore["config"]

# cost max for objective normalization
    if cost["UB"] != "unbounded":
        maxCost = cost["UB"]
    else:
        maxCost = 300000000.0

    optReqs = {
        "id": mor["id"],
        "description": mor["description"],
        "minMax": "max",
        "sensor_metrics": {
            "total": {
                "no_units": {
                    "minMax": "min",
                    "min": 0,
                    "max": 1000,
                    "UB": "unbounded",
                    "obj_weight": 0.0
                },
                "cost": {
                    "minMax": "min",
                    "min": 0,
                    "max": maxCost,
                    "UB": cost["UB"],
                    "obj_weight": cost["obj_weight"]
                }
            },
            "per_sensor": {
                s : {
                    "no_units": {
                        "UB": perSensor[s]["no_units"]["UB"]
                    },
                    "ppu": inputVarCore["sensors"][s]["ppu"],
                    "type": inputVarCore["sensors"][s]["type"],
                    "make": inputVarCore["sensors"][s]["make"],
                    "model": inputVarCore["sensors"][s]["model"]
                }
                for s in inputVarCore["sensors"]
            }

        },
        "coverage_areas_metrics": {
            "area_of_interest": {
                tt: {
                    "for_all_sensors": {
                        bc: {
                            "coverage_percentage": {
                                "minMax": "max",
                                "min": 0.0,
                                "max": 1.0,
                                "LB": performanceMetrics["roi"][tt]["for_all_sensors"][bc]["coverage_percentage"]["LB"],
                                "obj_weight": performanceMetrics["roi"][tt]["for_all_sensors"][bc]["coverage_percentage"]["obj_weight"],

                            },
                            "avg_redundancy": {"LB": "unbounded"}
                        }
                        for bc in config["binary_classifications"]
                    }
                }
                for tt in config["target_types"]
            },
            "by_coverage_area": {
                ca: {
                    tt: {
                        "for_all_sensors": {
                            bc: {
                                "coverage_percentage": {
                                    "minMax": "max",
                                    "min": 0.0,
                                    "max": 1.0,
                                    "LB": performanceMetrics["by_aop"][ca][tt]["for_all_sensors"][bc]["coverage_percentage"]["LB"],
                                    "obj_weight": performanceMetrics["by_aop"][ca][tt]["for_all_sensors"][bc]["coverage_percentage"]["obj_weight"],
                                },
                                "avg_redundancy": {"LB": "unbounded"}
                            }
                            for bc in inputVarCore["config"]["binary_classifications"]
                       }
                   }
                   for tt in config["target_types"]
              }
              for ca in inputVarCore["coverage_areas"]
           }
        },
        "protected_area_metrics": {
            "all_protected_areas": {
                "per_target_type": {
                    tt: {
                        "reaction_time_LB": 0.0,
                        "for_all_sensors": {
                            bc: {
                                "Pd_before_reaction_time": {
                                    "LB": 0.0
                                },
                                "lgPnd_UB": {
                                    "minMax": "min",
                                    "min": "unbounded",
                                    "max": 0.0,
                                    "obj_weight": 0.0
                                }
                            }
                            for bc in config["binary_classifications"]
                        }
                    }
                    for tt in config["target_types"]
                }
            },
            "by_protected_area": {}
        }
    }

    return optReqs

# end of wrapOptReqs
# ----------------------------------------------------------

def muscatSensorMetrics(muscatInputVarCore, muscatInputInstance):
    verifyInputVar(muscatInputVarCore)
    verifyInputInstance(muscatInputInstance, muscatInputVarCore)
    inputVarCore = wrapInputVarCore(muscatInputVarCore)
    inputInstance = wrapInputInstance(muscatInputInstance, inputVarCore)
    output = sa.sensorMetrics(inputVarCore, inputInstance)
    return output

def muscatOptiSensors(muscatInputVarCore, muscatOptReqs, muscatInputInstance):
    verifyInputVar(muscatInputVarCore)
    verifyOptReqs(muscatOptReqs, muscatInputVarCore)
    verifyInputInstance(muscatInputInstance, muscatInputVarCore)
    inputVarCore = wrapInputVarCore(muscatInputVarCore)
    inputInstance = wrapInputInstance(muscatInputInstance, inputVarCore)
    optReqs = wrapOptReqs(muscatOptReqs, inputVarCore)
    output = sa.optiSensors(inputVarCore, optReqs, inputInstance)

def muscatOptiSensorMetrics(muscatInputVarCore, muscatOptReqs, muscatInputInstance):
    verifyInputVar(muscatInputVarCore)
    verifyOptReqs(muscatOptReqs, muscatInputVarCore)
    verifyInputInstance(muscatInputInstance, muscatInputVarCore)
    inputVarCore = wrapInputVarCore(muscatInputVarCore)
    inputInstance = wrapInputInstance(muscatInputInstance, inputVarCore)
    optReqs = wrapOptReqs(muscatOptReqs, inputVarCore)
    answer = sa.optiSensors(inputVarCore, optReqs, inputInstance)
    return sa.sensorMetricsFromOpt(answer)