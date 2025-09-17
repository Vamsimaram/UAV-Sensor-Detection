# tabs/detection_probability.py

import streamlit as st
import folium
from streamlit_folium import folium_static
import numpy as np
import pandas as pd
import json
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.colors import LinearSegmentedColormap
import zipfile
import os
from datetime import datetime

# Import from map_utils
from map_utils import is_point_in_rectangle, is_point_in_polygon


def calculate_detection_probability(sensor_type, distance_km, sensor_specs, uav_altitude_km, uav_speed_kmh):
    """
    Calculate the probability of detection based on sensor type, distance, and UAV parameters
    
    Parameters:
    -----------
    sensor_type : str
        Type of sensor (Radar, RF, LiDAR)
    distance_km : float
        Distance from sensor to the grid cell center in kilometers
    sensor_specs : dict
        Specifications of the sensor
    uav_altitude_km : float
        Altitude of the UAV in kilometers
    uav_speed_kmh : float
        Speed of the UAV in km/h
        
    Returns:
    --------
    float : Probability of detection (0.0 to 1.0)
    """
    # Get sensor detection range - handle both km and m units
    max_range_km = sensor_specs.get("detection_range", 10.0)
    
    # If detection_range seems to be in meters, convert to km
    if max_range_km > 100:
        max_range_km = max_range_km / 1000.0
    
    # If distance > max_range, probability is 0
    if distance_km > max_range_km:
        return 0.0
    
    # Base probability calculation using exponential decay
    base_prob = math.exp(-(distance_km / max_range_km)**2)
    
    # Apply sensor type specific modifiers
    stype = sensor_type.lower()
    if stype == "radar":
        altitude_factor = max(0.3, 1.0 - 0.1 * (uav_altitude_km / 1.0))
        speed_factor = min(1.2, 0.9 + 0.1 * min(uav_speed_kmh / 72, 1.0))
        rcs_factor = 0.95
        final_prob = base_prob * altitude_factor * speed_factor * rcs_factor

    elif stype == "rf":
        signal_factor = max(0.3, 1.0 - 0.3 * (distance_km / max_range_km)**1.5)
        altitude_factor = max(0.3, 1.0 - 0.05 * (uav_altitude_km / 0.5))
        speed_factor = 1.0
        final_prob = base_prob * signal_factor * speed_factor * altitude_factor

    elif stype == "lidar":
        range_factor = max(0.3, 1.0 - 0.5 * (distance_km / max_range_km)**1.2)
        speed_factor = max(0.3, 1.0 - 0.2 * min(uav_speed_kmh / 108, 1.0))
        altitude_factor = max(0.3, 1.0 - 0.2 * (uav_altitude_km / 0.3))
        final_prob = base_prob * range_factor * speed_factor * altitude_factor

    else:
        final_prob = base_prob
    
    # Add small random variation (kept small for stability)
    randomness = max(0, min(1, np.random.normal(0, 0.02) + final_prob))
    return round(max(0.0, min(1.0, randomness)), 3)


def process_sensor_specifications_from_json(sensor_data_json):
    """
    Process sensor specifications from the nested JSON structure
    
    Parameters:
    -----------
    sensor_data_json : dict
        The loaded sensor data JSON
        
    Returns:
    --------
    list : Flattened list of sensor specifications
    """
    processed_specs = []
    
    if 'sensors' in sensor_data_json:
        for sensor_category in sensor_data_json['sensors']:
            sensor_type = sensor_category.get('sensor_type', 'Unknown')
            for param in sensor_category.get('parameters', []):
                spec = {
                    'type': sensor_type,
                    'model': param.get('model', 'Unknown'),
                    'manufacturer': param.get('manufacturer', 'Unknown'),
                    'detection_range': param.get('detection_range', 1.0),  # km (or meters, auto-convert later)
                    'response_time': param.get('response_time', 1.0),
                    'price_per_unit': param.get('price_per_unit', 50000),
                    'description': param.get('description', ''),
                    'sensor_specifications': param.get('sensor_specifications', {})
                }
                processed_specs.append(spec)
    return processed_specs


# FIXED: Updated preprocess_detection_probability function with unique sensor keys
def preprocess_detection_probability(grid_data, sensor_locations, sensor_specifications, uav_specs):
    """
    Preprocess and calculate detection probabilities for each grid cell
    FIXED: Uses unique keys for each sensor specification to prevent overwrites
    """
    results = {}
    if not sensor_locations or not sensor_specifications:
        return results
    
    # Extract UAV parameters
    uav_altitude_km = uav_specs.get("altitude", 0.1)
    uav_speed_kmh = uav_specs.get("speed", 36)
    
    for grid_cell in grid_data:
        grid_id = grid_cell["grid_id"]
        cell_center = grid_cell["center"]  # [lat, lng]
        
        cell_data = {
            "grid_id": grid_id,
            "center": cell_center,
            "sensor_locations": {}
        }
        
        # Calculate probabilities for each sensor location
        for i, sensor in enumerate(sensor_locations):
            sensor_pos = [sensor["lat"], sensor["lng"]]
            sensor_name = sensor.get("name", f"Sensor_{i+1}")
            
            # Haversine distance (km)
            lat1, lng1 = sensor_pos
            lat2, lng2 = cell_center
            lat1_rad = math.radians(lat1)
            lng1_rad = math.radians(lng1)
            lat2_rad = math.radians(lat2)
            lng2_rad = math.radians(lng2)
            earth_radius_km = 6371.0
            dlat = lat2_rad - lat1_rad
            dlng = lng2_rad - lng1_rad
            a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance_km = earth_radius_km * c
            
            sensor_location_data = {
                "position": sensor_pos,
                "distance": distance_km,
                "sensor_types": {}
            }
            
            # Compute probabilities for each sensor spec
            for j, sensor_spec in enumerate(sensor_specifications):
                sensor_type = sensor_spec.get("type", "Unknown")
                model = sensor_spec.get("model", "Unknown")
                manufacturer = sensor_spec.get("manufacturer", "Unknown")
                
                prob = calculate_detection_probability(
                    sensor_type,
                    distance_km,
                    sensor_spec,
                    uav_altitude_km,
                    uav_speed_kmh
                )
                
                # Unique key to prevent clashes
                unique_sensor_key = f"{sensor_type.lower().strip()}_{model.strip()}_{manufacturer.strip()}".replace(" ", "_").replace("-", "_").replace(".", "")
                
                # Helpful fallback keys
                type_key = sensor_type.strip().lower()
                model_key = f"{type_key}_{model.strip().replace(' ', '_').replace('-', '_').replace('.', '')}"
                
                sensor_data = {
                    "probability": prob,
                    "model": model,
                    "manufacturer": manufacturer,
                    "original_type": sensor_type,
                    "spec_index": j,
                    "unique_key": unique_sensor_key
                }
                
                # Store with multiple keys
                sensor_location_data["sensor_types"][unique_sensor_key] = sensor_data
                sensor_location_data["sensor_types"][model_key] = sensor_data
                if type_key not in sensor_location_data["sensor_types"]:
                    sensor_location_data["sensor_types"][type_key] = sensor_data
                if sensor_type != type_key:
                    sensor_location_data["sensor_types"][sensor_type] = sensor_data
            
            cell_data["sensor_locations"][sensor_name] = sensor_location_data
        
        results[grid_id] = cell_data
    
    return results


def create_combined_json_output(
    all_uav_results,
    sensor_locations,            # from st.session_state.potential_locations (user-named points)
    sensor_specifications,
    uav_specifications_list,
    boundary_type,
    boundary_points,
    sw_corner,
    ne_corner,
    grid_size_degrees,
    protected_areas=None
):
    """
    Build the main MUSCAT input (core) JSON and separate display-map JSON.

    Returns:
        main_json_string          -> contents for muscat_input_var_core.json
        probability_files         -> dict: {filename -> json_string} for detection arrays
        display_map_data_string   -> contents for display_map_data.json
    """
    import json
    import numpy as np

    if protected_areas is None:
        protected_areas = []

    # ---------------------------
    # Grid geometry (rows/cols)
    # ---------------------------
    if sw_corner and ne_corner:
        center_lat = (sw_corner[0] + ne_corner[0]) / 2.0
    else:
        lats = [p[0] for p in boundary_points]
        center_lat = sum(lats) / len(lats)

    lat_grid_size = grid_size_degrees
    lng_grid_size = grid_size_degrees / np.cos(np.radians(center_lat))

    if boundary_type == "rectangle" and sw_corner and ne_corner:
        width_degrees = ne_corner[1] - sw_corner[1]
        height_degrees = ne_corner[0] - sw_corner[0]
        num_rows = int(np.ceil(height_degrees / lat_grid_size))
        num_cols = int(np.ceil(width_degrees / lng_grid_size))
    else:
        lats = [p[0] for p in boundary_points]
        lngs = [p[1] for p in boundary_points]
        width_degrees = max(lngs) - min(lngs)
        height_degrees = max(lats) - min(lats)
        num_rows = int(np.ceil(height_degrees / lat_grid_size))
        num_cols = int(np.ceil(width_degrees / lng_grid_size))

    # ---------------------------
    # Target types & config
    # ---------------------------
    target_types = {}
    for uav_spec in uav_specifications_list:
        uav_name = uav_spec.get("name", uav_spec.get("id", "UAV"))
        altitude_m = uav_spec.get("altitude", 0.1) * 1000.0
        speed_ms = uav_spec.get("speed", 36.0) / 3.6
        target_types[uav_name] = {"min_altitude": altitude_m, "max_velocity": speed_ms}

    main_json = {
        "sensor_types": [],
        "sensors": {},
        "coverage_areas": {},
        "title": "Sensor Detection Analysis",
        "config": {
            "y_length": num_rows,
            "long_lat_unit": "degree",
            "x_length": num_cols,
            "sq_side_units": "long_lat_degree",
            "square_side": grid_size_degrees,
            "time_unit": "hour",
            "earth_radius": 6371000,
            "binary_classifications": {
                "target_detection": {"Pd_TH": 0.6},
                "foe_or_friend": {"Pd_TH": 0.6}
            },
            "distance_unit": "m",
            "south_west_corner": {
                "long": sw_corner[1] if sw_corner else min([p[1] for p in boundary_points]),
                "lat":  sw_corner[0] if sw_corner else min([p[0] for p in boundary_points])
            },
            "target_types": target_types
        },
        "area_of_interest": []
    }

    probability_files = {}
    grid_cells_info = {}

    # ---------------------------
    # Grid cells (use first UAV's map as scaffold)
    # ---------------------------
    if all_uav_results:
        first_uav_id = list(all_uav_results.keys())[0]
        grid_probabilities = all_uav_results[first_uav_id]["grid_probabilities"]
        for grid_id, data in grid_probabilities.items():
            if grid_id.startswith("r") and "c" in grid_id:
                try:
                    parts = grid_id[1:].split("c")
                    row = int(parts[0])
                    col = int(parts[1])
                    linear_index = row * num_cols + col
                    grid_cells_info[linear_index] = {
                        "grid_id": grid_id,
                        "center": data["center"],
                        "sw_corner": data.get("sw_corner", data["center"]),
                    }
                except (ValueError, IndexError):
                    continue

    # Area of interest: store SW corners as [lng, lat]
    if grid_cells_info:
        main_json["area_of_interest"] = [
            [cell["sw_corner"][1], cell["sw_corner"][0]]  # [lng, lat]
            for _, cell in sorted(grid_cells_info.items())
        ]

    # ---------------------------
    # AOI square for display_map_data (lat,lng)
    # ---------------------------
    aoi_square = {}
    if boundary_type == "rectangle" and sw_corner and ne_corner:
        lat_sw, lng_sw = sw_corner
        lat_ne, lng_ne = ne_corner
        aoi_square = {
            "sw": [lat_sw, lng_sw],
            "se": [lat_sw, lng_ne],
            "ne": [lat_ne, lng_ne],
            "nw": [lat_ne, lng_sw],
        }
    elif boundary_type == "polygon" and boundary_points:
        lats = [p[0] for p in boundary_points]
        lngs = [p[1] for p in boundary_points]
        lat_sw, lng_sw = min(lats), min(lngs)
        lat_ne, lng_ne = max(lats), max(lngs)
        aoi_square = {
            "sw": [lat_sw, lng_sw],
            "se": [lat_sw, lng_ne],
            "ne": [lat_ne, lng_ne],
            "nw": [lat_ne, lng_sw],
        }

    # ---------------------------
    # Protected areas
    # ---------------------------
    try:
        # prefer explicit arg; fall back to session for safety
        from streamlit import session_state as _ss
        protected_areas_list = protected_areas or _ss.get("protected_areas", [])
    except Exception:
        protected_areas_list = protected_areas or []

    # Raw coords for display_map_data
    protected_areas_coords = {}
    for area in protected_areas_list:
        area_name = area["name"]
        pts = [{"lat": lat, "long": lng} for lat, lng in area["points"]]
        protected_areas_coords[area_name] = pts

    # Indices per area (for core "coverage_areas")
    if protected_areas_list and grid_cells_info:
        from map_utils import is_point_in_polygon
        for area in protected_areas_list:
            area_name = area["name"]
            area_points = area["points"]
            cells_in_area = []
            for linear_index, cell_info in grid_cells_info.items():
                if is_point_in_polygon(cell_info["center"], area_points):
                    cells_in_area.append(linear_index)
            if cells_in_area:
                main_json["coverage_areas"][area_name] = {"area": sorted(cells_in_area)}
    if not main_json["coverage_areas"]:
        all_grid_cells = sorted(list(grid_cells_info.keys()))
        main_json["coverage_areas"]["default_area"] = {"area": all_grid_cells}

    # ---------------------------
    # Sensors (core) + detection files
    # ---------------------------
    main_json["sensor_types"] = list(set(spec["type"].lower() for spec in sensor_specifications))

    # lightweight sensor->locs (by model name) for display_map_data (optional)
    display_sensors_map = {}

    for spec_idx, sensor_spec in enumerate(sensor_specifications):
        sensor_type = sensor_spec["type"].lower().strip()
        sensor_model = sensor_spec["model"].strip()
        sensor_manufacturer = sensor_spec.get("manufacturer", "").strip()

        exact_sensor_name = f"{sensor_model}_{sensor_manufacturer}".replace(" ", "_").replace("-", "_").replace(".", "")
        if len(exact_sensor_name) > 50:
            exact_sensor_name = sensor_model.replace(" ", "_").replace("-", "_").replace(".", "")
        if exact_sensor_name and exact_sensor_name[0].isdigit():
            exact_sensor_name = f"sensor_{exact_sensor_name}"

        unique_sensor_key = f"{sensor_type}_{sensor_model}_{sensor_manufacturer}".replace(" ", "_").replace("-", "_").replace(".", "")

        main_json["sensors"][exact_sensor_name] = {
            "possible_locs": [],
            "optional_params": "Sensor parameters for detection model",
            "ppu": sensor_spec.get("price_per_unit", 500000),
            "model": sensor_model,
            "detection_period": 0.01,
            "type": sensor_type,
            "make": sensor_manufacturer
        }

        display_sensors_map[exact_sensor_name] = []

        # Each possible location from st.session_state.potential_locations
        for j, location in enumerate(sensor_locations):
            coverage_metrics = {}
            for uav_spec in uav_specifications_list:
                uav_name = uav_spec.get("name", uav_spec.get("id", "UAV"))
                uav_id = uav_spec.get("id", uav_name)
                detection_filename = f"{exact_sensor_name}_loc{j+1}_{uav_name}_detection.json"

                coverage_metrics[uav_name] = {
                    "target_detection": {"Pd": {"@file": detection_filename}, "Pfa": "tbd"},
                    "foe_or_friend": {"Pd": {"@file": detection_filename}},
                }

                # Build detection probabilities vector (grid-linearized order)
                detection_probs = []
                all_grid_cells_idxs = sorted(list(grid_cells_info.keys()))

                if uav_id in all_uav_results:
                    grid_probabilities = all_uav_results[uav_id]["grid_probabilities"]
                    prob_by_index = {}

                    for grid_id, data in grid_probabilities.items():
                        if grid_id.startswith("r") and "c" in grid_id:
                            try:
                                parts = grid_id[1:].split("c")
                                row = int(parts[0]); col = int(parts[1])
                                linear_index = row * num_cols + col
                                prob = 0.0

                                # match by user sensor location name
                                sensor_name_at_loc = location.get("name", f"Sensor_{j+1}")
                                if sensor_name_at_loc in data["sensor_locations"]:
                                    sensor_data = data["sensor_locations"][sensor_name_at_loc]
                                    found = False

                                    # Try unique key
                                    if unique_sensor_key in sensor_data["sensor_types"]:
                                        prob = sensor_data["sensor_types"][unique_sensor_key]["probability"]; found = True
                                    # Try model key
                                    if not found:
                                        model_key = f"{sensor_type}_{sensor_model.strip().replace(' ', '_').replace('-', '_').replace('.', '')}"
                                        if model_key in sensor_data["sensor_types"]:
                                            prob = sensor_data["sensor_types"][model_key]["probability"]; found = True
                                    # Try type + model verification
                                    if not found and sensor_type in sensor_data["sensor_types"]:
                                        type_data = sensor_data["sensor_types"][sensor_type]
                                        if type_data["model"].strip() == sensor_model.strip():
                                            prob = type_data["probability"]; found = True
                                    # Fallback: spec index match
                                    if not found:
                                        for _, type_data in sensor_data["sensor_types"].items():
                                            if (type_data.get("model","").strip() == sensor_model.strip()
                                                and type_data.get("spec_index") == spec_idx):
                                                prob = type_data["probability"]; break

                                prob_by_index[linear_index] = prob
                            except (ValueError, IndexError):
                                continue

                    for linear_index in all_grid_cells_idxs:
                        detection_probs.append(prob_by_index.get(linear_index, 0.0))

                probability_files[detection_filename] = "[\n" + ",\n".join([f"  {p}" for p in detection_probs]) + "\n]"

            # record in core
            main_json["sensors"][exact_sensor_name]["possible_locs"].append({
                "coverage_metrics": coverage_metrics,
                "long": location["lng"],
                "lat": location["lat"]
            })

            # lightweight (for display_model mapping - optional/UI)
            display_sensors_map[exact_sensor_name].append({
                "lat": location["lat"],
                "long": location["lng"]
            })

    # ---------------------------
    # Serialize core JSON
    # ---------------------------
    main_json_string = json.dumps(main_json, indent=2)

    # ---------------------------
    # Build display_map_data.json
    #   - AOI square, protected areas
    #   - model-based list & mapping (optional)
    #   - USER-ENTERED named sensor points (the important bit)
    # ---------------------------
    # user-named sensors from Possible Sensor Locations tab
    user_sensor_points = []
    user_sensor_map = {}
    for loc in sensor_locations:
        entry = {
            "name": loc.get("name") or "",
            "lat": loc["lat"],
            "long": loc["lng"]
        }
        user_sensor_points.append(entry)
        if entry["name"]:
            user_sensor_map[entry["name"]] = {"lat": entry["lat"], "long": entry["long"]}

    display_map_data = {
        "square_coordinates": aoi_square,
        "protected_areas_coordinates": protected_areas_coords,
        # optional convenience fields (from models/cores)
        "sensor_names": sorted(list(main_json["sensors"].keys())),
        "sensors": display_sensors_map,            # model-sensor -> [{lat,long}, ...]
        # the key fields you asked for (what the user actually named & placed)
        "user_sensor_points": user_sensor_points,  # list of {name,lat,long}
        "user_sensor_map": user_sensor_map         # dict: name -> {lat,long}
    }
    display_map_data_string = json.dumps(display_map_data, indent=2)

    return main_json_string, probability_files, display_map_data_string



def detection_probability_tab():
    """
    Preprocess Detection Probability Tab with support for multiple UAV specifications,
    generating both main JSON and separate probability files in a ZIP.
    Now also writes display_map_data.json with square_coordinates and protected_areas_coordinates.
    """
    st.header("Detection Probability Analysis")

    # ==== If we've already locked, still let user download again and jump to Display ====
    if st.session_state.get('detection_locked', False):
        st.info("Detection has been run. Inputs are locked. See the Display tab for the results.")

        zip_bytes = st.session_state.get('zip_data')
        if zip_bytes:
            fname = st.session_state.get('detection_zip_filename') \
                    or os.path.basename(st.session_state.get('detection_zip_path', 'muscat_detection.zip'))
            st.download_button(
                label="Download All Detection Data (ZIP)",
                data=zip_bytes,
                file_name=fname,
                mime="application/zip",
                help="Main JSON, display_map_data.json, and detection probability files"
            )

        cols = st.columns(2)
        with cols[0]:
            if st.button("Open Display tab"):
                st.session_state.active_tab = "Display"  # if your app uses this for tab routing
                st.rerun()
        with cols[1]:
            if st.button("Unlock to re-run"):
                st.session_state.detection_locked = False
                st.rerun()
        return

    # ==== Initial preconditions (only when not locked) ====
    if not st.session_state.location_selected:
        st.warning("Please select a location in the initial screen first.")
        if st.button("Return to Location Selection"):
            st.session_state.location_selected = False
            st.rerun()
        return

    if not st.session_state.area_selected:
        st.warning("Please select an area of interest in the Map & Selection tab first.")
        return

    if not st.session_state.potential_locations:
        st.warning("Please place sensors in the Possible Sensor Placement tab first.")
        return

    if not st.session_state.get('uav_specifications_list'):
        st.warning("Please add at least one UAV configuration in the sidebar.")
        return

    st.info(f"Found {len(st.session_state.uav_specifications_list)} UAV configuration(s) to analyze.")

    run_button = st.button("Run Detection Probability", type="primary")

    # If a run already happened (but not locked yet), offer download immediately
    if st.session_state.get('detection_prob_calculated', False) and st.session_state.get('zip_data'):
        st.success("Detection probability analysis completed!")
        downloaded_now = st.download_button(
            label="Download All Detection Data (ZIP)",
            data=st.session_state.zip_data,
            file_name=st.session_state.get('detection_zip_filename', 'muscat_detection.zip'),
            mime="application/zip",
            help="Download main JSON, display_map_data.json, and detection probability files in a ZIP archive"
        )
        # 👉 Lock only after download
        if downloaded_now:
            st.session_state.detection_locked = True
            st.success("ZIP downloaded. Inputs are now locked. Open the Display tab to view results.")
            st.rerun()
        return

    # ==== Fresh run ====
    if run_button:
        with st.spinner("Calculating detection probabilities for all UAV configurations..."):
            if st.session_state.grid_enabled and "grid_size_degrees" in st.session_state:
                sw_corner = st.session_state.sw_corner
                ne_corner = st.session_state.ne_corner
                grid_size_degrees = st.session_state.grid_size_degrees

                center_lat = (sw_corner[0] + ne_corner[0]) / 2
                lat_grid_size = grid_size_degrees
                lng_grid_size = grid_size_degrees / np.cos(np.radians(center_lat))

                width_degrees = ne_corner[1] - sw_corner[1]
                height_degrees = ne_corner[0] - sw_corner[0]

                num_rows = int(np.ceil(height_degrees / lat_grid_size))
                num_cols = int(np.ceil(width_degrees / lng_grid_size))

                grid_data = []
                for i in range(num_rows):
                    for j in range(num_cols):
                        lat_sw = sw_corner[0] + i * lat_grid_size
                        lng_sw = sw_corner[1] + j * lng_grid_size
                        lat_ne = lat_sw + lat_grid_size
                        lng_ne = lng_sw + lng_grid_size
                        center_lat_cell = (lat_sw + lat_ne) / 2
                        center_lng_cell = (lng_sw + lng_ne) / 2

                        if st.session_state.boundary_type == "rectangle":
                            is_inside = (sw_corner[0] <= center_lat_cell <= ne_corner[0] and 
                                         sw_corner[1] <= center_lng_cell <= ne_corner[1])
                        elif st.session_state.boundary_type == "polygon":
                            is_inside = is_point_in_polygon(
                                [center_lat_cell, center_lng_cell],
                                st.session_state.boundary_points
                            )
                        else:
                            is_inside = True

                        if is_inside:
                            grid_cell = {
                                "grid_id": f"r{i}c{j}",
                                "row": i,
                                "col": j,
                                "sw_corner": [lat_sw, lng_sw],
                                "ne_corner": [lat_ne, lng_ne],
                                "center": [center_lat_cell, center_lng_cell],
                                "size_degrees": grid_size_degrees
                            }
                            grid_data.append(grid_cell)

                # Calculate detection probabilities for each UAV configuration
                all_uav_results = {}
                for uav_spec in st.session_state.uav_specifications_list:
                    uav_specs = {"altitude": uav_spec["altitude"], "speed": uav_spec["speed"]}
                    grid_probabilities = preprocess_detection_probability(
                        grid_data,
                        st.session_state.potential_locations,
                        st.session_state.sensor_specifications,
                        uav_specs
                    )
                    # add SW corner to each grid cell in results
                    for grid_id, prob_data in grid_probabilities.items():
                        for grid_cell in grid_data:
                            if grid_cell["grid_id"] == grid_id:
                                prob_data["sw_corner"] = grid_cell["sw_corner"]
                                break
                    all_uav_results[uav_spec["id"]] = {
                        "uav_spec": uav_spec,
                        "grid_probabilities": grid_probabilities
                    }

                # Build outputs (main + detection files + display_map_data)
                main_json, probability_files, display_map_data_json = create_combined_json_output(
                    all_uav_results,
                    st.session_state.potential_locations,
                    st.session_state.sensor_specifications,
                    st.session_state.uav_specifications_list,
                    st.session_state.boundary_type,
                    st.session_state.boundary_points,
                    st.session_state.sw_corner,
                    st.session_state.ne_corner,
                    grid_size_degrees,
                    protected_areas=st.session_state.get('protected_areas', [])
                )

                # Save to session
                st.session_state.main_json = main_json
                st.session_state.probability_files = probability_files
                st.session_state.display_map_data_json = display_map_data_json

                # Hydrate parsed dicts for the Display tab
                try:
                    st.session_state["muscat_input_var_core"] = json.loads(main_json)
                except Exception:
                    pass
                try:
                    st.session_state["display_map_data"] = json.loads(display_map_data_json)
                except Exception:
                    pass

                # Mark scenario available (but DO NOT lock yet)
                st.session_state["scenario_loaded"] = True
                st.session_state["detection_prob_calculated"] = True

                # Create a zip file containing all JSON files
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Main JSON
                    zipf.writestr("muscat_input_var_core.json", main_json)
                    # Display map data JSON
                    zipf.writestr("display_map_data.json", display_map_data_json)
                    # All detection probability files
                    for filename, content in probability_files.items():
                        zipf.writestr(filename, content)

                zip_buffer.seek(0)
                st.session_state.zip_data = zip_buffer.getvalue()

                # Persist to disk (optional convenience)
                output_dir = "detection_outputs"
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_filename = f"muscat_input_var_core_{timestamp}.zip"
                zip_path = os.path.join(output_dir, zip_filename)
                with open(zip_path, 'wb') as f:
                    f.write(st.session_state.zip_data)
                st.session_state.detection_zip_path = zip_path
                st.session_state.detection_zip_filename = zip_filename

                # >>> DO NOT set detection_locked here. We lock only after download.

            else:
                st.error("Please enable and configure the grid in the Map & Selection tab first.")
                return

        # Show completion + download button (and lock only after click)
        if st.session_state.get('detection_prob_calculated', False) and st.session_state.get('zip_data'):
            st.success("Detection probability analysis completed!")
            downloaded_now = st.download_button(
                label="Download All Detection Data (ZIP)",
                data=st.session_state.zip_data,
                file_name=st.session_state.get('detection_zip_filename', 'muscat_detection.zip'),
                mime="application/zip",
                help="Download main JSON, display_map_data.json, and detection probability files in a ZIP archive"
            )
            if downloaded_now:
                st.session_state.detection_locked = True
                st.success("ZIP downloaded. Inputs are now locked. Open the Display tab to view results.")
                st.rerun()
