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
    
    # Check if detection_range is in meters (convert to km)
    if max_range_km > 100:  # Assume values > 100 are in meters
        max_range_km = max_range_km / 1000.0
    
    # print(f"DEBUG: Sensor {sensor_type}, Distance: {distance_km:.3f}km, Max Range: {max_range_km:.3f}km")
    
    # If distance > max_range, probability is 0
    if distance_km > max_range_km:
        # print(f"DEBUG: Distance {distance_km:.3f} > Range {max_range_km:.3f}, returning 0")
        return 0.0
    
    # Base probability calculation using exponential decay
    base_prob = math.exp(-(distance_km / max_range_km)**2)
    # print(f"DEBUG: Base probability: {base_prob:.3f}")
    
    # Apply sensor type specific modifiers
    if sensor_type.lower() == "radar":
        # Radar performance factors
        altitude_factor = max(0.3, 1.0 - 0.1 * (uav_altitude_km / 1.0))
        speed_factor = min(1.2, 0.9 + 0.1 * min(uav_speed_kmh / 72, 1.0))
        rcs_factor = 0.95
        
        final_prob = base_prob * altitude_factor * speed_factor * rcs_factor
    
    elif sensor_type.lower() == "rf":
        # RF detection factors
        signal_factor = max(0.3, 1.0 - 0.3 * (distance_km / max_range_km)**1.5)
        altitude_factor = max(0.3, 1.0 - 0.05 * (uav_altitude_km / 0.5))
        speed_factor = 1.0
        
        final_prob = base_prob * signal_factor * speed_factor * altitude_factor
    
    elif sensor_type.lower() == "lidar":
        # LiDAR factors
        range_factor = max(0.3, 1.0 - 0.5 * (distance_km / max_range_km)**1.2)
        speed_factor = max(0.3, 1.0 - 0.2 * min(uav_speed_kmh / 108, 1.0))
        altitude_factor = max(0.3, 1.0 - 0.2 * (uav_altitude_km / 0.3))
        
        final_prob = base_prob * range_factor * speed_factor * altitude_factor
    
    else:
        final_prob = base_prob
    
    # Add small random variation (reduce randomness for debugging)
    randomness = max(0, min(1, np.random.normal(0, 0.02) + final_prob))
    
    result = round(max(0.0, min(1.0, randomness)), 3)
    # print(f"DEBUG: Final probability: {result}")
    return result

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
                # Create a flattened specification
                spec = {
                    'type': sensor_type,
                    'model': param.get('model', 'Unknown'),
                    'manufacturer': param.get('manufacturer', 'Unknown'),
                    'detection_range': param.get('detection_range', 1.0),  # km
                    'response_time': param.get('response_time', 1.0),
                    'price_per_unit': param.get('price_per_unit', 50000),
                    'description': param.get('description', ''),
                    'sensor_specifications': param.get('sensor_specifications', {})
                }
                processed_specs.append(spec)
    
    # print(f"DEBUG: Processed {len(processed_specs)} sensor specifications")
    # for spec in processed_specs:
    #     print(f"DEBUG: {spec['type']} - {spec['model']} (range: {spec['detection_range']}km)")
    
    return processed_specs

# FIXED: Updated preprocess_detection_probability function with unique sensor keys
def preprocess_detection_probability(grid_data, sensor_locations, sensor_specifications, uav_specs):
    """
    Preprocess and calculate detection probabilities for each grid cell
    FIXED: Uses unique keys for each sensor specification to prevent overwrites
    """
    results = {}
    
    if not sensor_locations or not sensor_specifications:
        # print("DEBUG: No sensor locations or specifications")
        return results
    
    # Extract UAV parameters
    uav_altitude_km = uav_specs.get("altitude", 0.1)
    uav_speed_kmh = uav_specs.get("speed", 36)
    
    # print(f"DEBUG: UAV specs - Altitude: {uav_altitude_km}km, Speed: {uav_speed_kmh}km/h")
    # print(f"DEBUG: Processing {len(grid_data)} grid cells")
    # print(f"DEBUG: Sensor locations: {len(sensor_locations)}")
    # print(f"DEBUG: Sensor specifications: {len(sensor_specifications)}")
    
    # Debug sensor specifications
    # for i, spec in enumerate(sensor_specifications):
    #     print(f"DEBUG: Spec {i}: {spec.get('type', 'Unknown')} - {spec.get('model', 'Unknown')}")
    
    # Process each grid cell
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
            
            # Calculate distance using Haversine formula
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
            
            # print(f"DEBUG: Grid {grid_id}, Sensor {sensor_name}, Distance: {distance_km:.3f}km")
            
            # FIXED: Calculate probabilities for each sensor specification with unique keys
            for j, sensor_spec in enumerate(sensor_specifications):
                sensor_type = sensor_spec.get("type", "Unknown")
                model = sensor_spec.get("model", "Unknown")
                manufacturer = sensor_spec.get("manufacturer", "Unknown")
                
                # print(f"DEBUG: Processing sensor type: {sensor_type}, model: {model}")
                
                # Calculate probability
                prob = calculate_detection_probability(
                    sensor_type,
                    distance_km,
                    sensor_spec,
                    uav_altitude_km,
                    uav_speed_kmh
                )
                
                # FIXED: Create unique key that combines type, model, and manufacturer
                # This prevents different sensors from overwriting each other
                unique_sensor_key = f"{sensor_type.lower().strip()}_{model.strip()}_{manufacturer.strip()}".replace(" ", "_").replace("-", "_").replace(".", "")
                
                # Also create simpler keys for backward compatibility and fallback matching
                type_key = sensor_type.strip().lower()
                model_key = f"{type_key}_{model.strip().replace(' ', '_').replace('-', '_').replace('.', '')}"
                
                # Store the data with all relevant information
                sensor_data = {
                    "probability": prob,
                    "model": model,
                    "manufacturer": manufacturer,
                    "original_type": sensor_type,
                    "spec_index": j,  # Track which specification this is
                    "unique_key": unique_sensor_key
                }
                
                # FIXED: Store with multiple keys to ensure we can find it later
                # Primary unique key (most specific)
                sensor_location_data["sensor_types"][unique_sensor_key] = sensor_data
                
                # Model-specific key (medium specificity)
                sensor_location_data["sensor_types"][model_key] = sensor_data
                
                # Type key (least specific, for fallback)
                # Only store if this is the first sensor of this type, or if it overwrites, ensure it's consistent
                if type_key not in sensor_location_data["sensor_types"]:
                    sensor_location_data["sensor_types"][type_key] = sensor_data
                
                # Original type for exact matching
                if sensor_type != type_key:
                    sensor_location_data["sensor_types"][sensor_type] = sensor_data
                
                # print(f"DEBUG: Stored probability {prob} with keys: {unique_sensor_key}, {model_key}, {type_key}")
            
            cell_data["sensor_locations"][sensor_name] = sensor_location_data
        
        results[grid_id] = cell_data
    
    # print(f"DEBUG: Completed processing, returning {len(results)} grid results")
    return results

# FIXED: Updated create_combined_json_output function with improved sensor matching
def create_combined_json_output(all_uav_results, sensor_locations, sensor_specifications, 
                            uav_specifications_list, boundary_type, boundary_points, 
                            sw_corner, ne_corner, grid_size_degrees):
    """
    Create combined output with improved sensor matching logic and exact sensor names
    FIXED: Properly matches sensors using unique keys to prevent data loss
    """
    import json
    import numpy as np
    
    # print(f"DEBUG: Starting JSON creation with {len(all_uav_results)} UAV results")
    
    # Calculate center latitude for calculations
    if sw_corner and ne_corner:
        center_lat = (sw_corner[0] + ne_corner[0]) / 2
    else:
        lats = [point[0] for point in boundary_points]
        center_lat = sum(lats) / len(lats)
    
    # Grid calculations
    lat_grid_size = grid_size_degrees
    lng_grid_size = grid_size_degrees / np.cos(np.radians(center_lat))
    
    if boundary_type == "rectangle" and sw_corner and ne_corner:
        width_degrees = ne_corner[1] - sw_corner[1]
        height_degrees = ne_corner[0] - sw_corner[0]
        num_rows = int(np.ceil(height_degrees / lat_grid_size))
        num_cols = int(np.ceil(width_degrees / lng_grid_size))
    else:
        lats = [point[0] for point in boundary_points]
        lngs = [point[1] for point in boundary_points]
        width_degrees = max(lngs) - min(lngs)
        height_degrees = max(lats) - min(lats)
        num_rows = int(np.ceil(height_degrees / lat_grid_size))
        num_cols = int(np.ceil(width_degrees / lng_grid_size))
    
    # print(f"DEBUG: Grid dimensions: {num_rows} x {num_cols} = {num_rows * num_cols} cells")
    
    # Create target_types dictionary
    target_types = {}
    for uav_spec in uav_specifications_list:
        uav_name = uav_spec.get("name", uav_spec.get("id", "UAV"))
        altitude_m = uav_spec.get("altitude", 0.1) * 1000
        speed_ms = uav_spec.get("speed", 36) / 3.6
        
        target_types[uav_name] = {
            "min_altitude": altitude_m,
            "max_velocity": speed_ms
        }
    
    # Create binary classifications
    binary_classifications = {}
    for uav_spec in uav_specifications_list:
        uav_name = uav_spec.get("name", uav_spec.get("id", "UAV"))
        binary_classifications[uav_name] = {"Pd_TH": 0.8}
    
    # Main JSON structure
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
                "target_detection": {
                    "Pd_TH": 0.8
                },
                "foe_or_friend": {
                    "Pd_TH": 0.8
                }
            },
            "distance_unit": "m",
            "south_west_corner": {
                "long": sw_corner[1] if sw_corner else min([p[1] for p in boundary_points]),
                "lat": sw_corner[0] if sw_corner else min([p[0] for p in boundary_points])
            },
            "target_types": target_types
        },
        "area_of_interest": []
    }
    
    probability_files = {}
    grid_cells_info = {}
    
    # Extract grid cells info with better error handling
    if all_uav_results:
        first_uav_id = list(all_uav_results.keys())[0]
        grid_probabilities = all_uav_results[first_uav_id]["grid_probabilities"]
        
        # print(f"DEBUG: Processing {len(grid_probabilities)} grid cells from UAV {first_uav_id}")
        
        for grid_id, data in grid_probabilities.items():
            if grid_id.startswith('r') and 'c' in grid_id:
                try:
                    parts = grid_id[1:].split('c')
                    row = int(parts[0])
                    col = int(parts[1])
                    linear_index = row * num_cols + col
                    grid_cells_info[linear_index] = {
                        'grid_id': grid_id,
                        'center': data['center'],
                        'sw_corner': data.get('sw_corner', data['center'])
                    }
                except (ValueError, IndexError) as e:
                    # print(f"WARNING: Could not parse grid_id {grid_id}: {e}")
                    continue
    
    # Area of interest
    area_of_interest_sw_corners = []
    if grid_cells_info:
        for linear_index, cell_info in grid_cells_info.items():
            cell_sw_corner = cell_info['sw_corner']
            # CHANGE: Swap lat/lng order to match working examples [lng, lat]
            area_of_interest_sw_corners.append([cell_sw_corner[1], cell_sw_corner[0]])

    main_json["area_of_interest"] = area_of_interest_sw_corners
    
    # Extract unique sensor types
    sensor_types = list(set(spec["type"].lower() for spec in sensor_specifications))
    main_json["sensor_types"] = sensor_types
    
    # print(f"DEBUG: Sensor types: {sensor_types}")
    
    # Process coverage areas (protected areas)
    import streamlit as st
    protected_areas = st.session_state.get('protected_areas', [])
    
    if protected_areas:
        from map_utils import is_point_in_polygon
        
        for area in protected_areas:
            area_name = area['name']
            area_points = area['points']
            
            cells_in_area = []
            for linear_index, cell_info in grid_cells_info.items():
                cell_center = cell_info['center']
                if is_point_in_polygon(cell_center, area_points):
                    cells_in_area.append(linear_index)
            
            cells_in_area.sort()
            if cells_in_area:
                main_json["coverage_areas"][area_name] = {
                    "area": cells_in_area
                }
    
    # Default coverage area if none defined
    if not main_json["coverage_areas"]:
        all_grid_cells = sorted(list(grid_cells_info.keys()))
        main_json["coverage_areas"]["default_area"] = {
            "area": all_grid_cells
        }
    
    # FIXED: Process each unique sensor specification with exact names and proper matching
    for spec_idx, sensor_spec in enumerate(sensor_specifications):
        sensor_type = sensor_spec["type"].lower().strip()
        sensor_model = sensor_spec["model"].strip()
        sensor_manufacturer = sensor_spec.get("manufacturer", "").strip()
        
        # Create exact sensor name using model and manufacturer
        # Format: "ModelName_Manufacturer" (replace spaces with underscores, remove special chars)
        exact_sensor_name = f"{sensor_model}_{sensor_manufacturer}".replace(" ", "_").replace("-", "_").replace(".", "")
        
        # If the exact name is too long or has issues, use just the model name
        if len(exact_sensor_name) > 50:
            exact_sensor_name = sensor_model.replace(" ", "_").replace("-", "_").replace(".", "")
        
        # Ensure the name doesn't start with a number (JSON key requirement)
        if exact_sensor_name[0].isdigit():
            exact_sensor_name = f"sensor_{exact_sensor_name}"
        
        # FIXED: Create the same unique key used in preprocess_detection_probability
        unique_sensor_key = f"{sensor_type}_{sensor_model}_{sensor_manufacturer}".replace(" ", "_").replace("-", "_").replace(".", "")
        
        # print(f"DEBUG: Processing sensor {exact_sensor_name} - {sensor_type} {sensor_model} (unique_key: {unique_sensor_key})")
        
        main_json["sensors"][exact_sensor_name] = {
            "possible_locs": [],
            "optional_params": "Sensor parameters for detection model",
            "ppu": sensor_spec.get("price_per_unit", 500000),
            "model": sensor_model,
            "detection_period": 0.01,
            "type": sensor_type,
            "make": sensor_manufacturer
        }
        
        # Process each sensor location
        for j, location in enumerate(sensor_locations):
            coverage_metrics = {}
            
            # Process each UAV specification
            for uav_spec in uav_specifications_list:
                uav_name = uav_spec.get("name", uav_spec.get("id", "UAV"))
                uav_id = uav_spec.get("id", uav_name)
                
                # Use exact sensor name in filename
                detection_filename = f"{exact_sensor_name}_loc{j+1}_{uav_name}_detection.json"
                
                coverage_metrics[uav_name] = {
                    "target_detection": {
                        "Pd": {"@file": detection_filename},
                        "Pfa": "tbd"
                    },
                    "foe_or_friend": {
                        "Pd": {"@file": detection_filename}
                    }
                }
                
                # Create detection probabilities data
                detection_probs = []
                all_grid_cells = sorted(list(grid_cells_info.keys()))
                
                # print(f"DEBUG: Creating detection file {detection_filename} for {len(all_grid_cells)} cells")
                
                if uav_id in all_uav_results:
                    grid_probabilities = all_uav_results[uav_id]["grid_probabilities"]
                    prob_by_index = {}
                    
                    # Extract probabilities for each grid cell
                    for grid_id, data in grid_probabilities.items():
                        if grid_id.startswith('r') and 'c' in grid_id:
                            try:
                                parts = grid_id[1:].split('c')
                                row = int(parts[0])
                                col = int(parts[1])
                                linear_index = row * num_cols + col
                                
                                # Find probability for this sensor
                                prob = 0.0
                                sensor_name = location.get("name", f"Sensor_{j+1}")
                                
                                if sensor_name in data["sensor_locations"]:
                                    sensor_data = data["sensor_locations"][sensor_name]
                                    
                                    # FIXED: Try multiple approaches to find the matching sensor type
                                    found = False
                                    
                                    # Approach 1: Try unique sensor key (most specific)
                                    if unique_sensor_key in sensor_data["sensor_types"]:
                                        type_data = sensor_data["sensor_types"][unique_sensor_key]
                                        prob = type_data["probability"]
                                        found = True
                                        # print(f"DEBUG: Found unique key match for {sensor_name}, {unique_sensor_key}: {prob}")
                                    
                                    # Approach 2: Try model-specific key
                                    if not found:
                                        model_key = f"{sensor_type}_{sensor_model.strip().replace(' ', '_').replace('-', '_').replace('.', '')}"
                                        if model_key in sensor_data["sensor_types"]:
                                            type_data = sensor_data["sensor_types"][model_key]
                                            prob = type_data["probability"]
                                            found = True
                                            # print(f"DEBUG: Found model key match for {sensor_name}, {model_key}: {prob}")
                                    
                                    # Approach 3: Try exact sensor type match with model verification
                                    if not found and sensor_type in sensor_data["sensor_types"]:
                                        type_data = sensor_data["sensor_types"][sensor_type]
                                        if type_data["model"].strip() == sensor_model.strip():
                                            prob = type_data["probability"]
                                            found = True
                                            # print(f"DEBUG: Found exact match for {sensor_name}, {sensor_type}, {sensor_model}: {prob}")
                                    
                                    # Approach 4: Try all available sensor types for this model (fallback)
                                    if not found:
                                        for available_type, type_data in sensor_data["sensor_types"].items():
                                            if (type_data["model"].strip() == sensor_model.strip() and 
                                                type_data.get("spec_index") == spec_idx):
                                                prob = type_data["probability"]
                                                found = True
                                                # print(f"DEBUG: Found spec index match for {sensor_name}, {available_type}, {sensor_model}: {prob}")
                                                break
                                    
                                    # if not found:
                                    #     print(f"WARNING: No probability found for {sensor_name}, type {sensor_type}, model {sensor_model}, unique_key {unique_sensor_key}")
                                    #     print(f"Available keys: {list(sensor_data['sensor_types'].keys())}")
                                
                                prob_by_index[linear_index] = prob
                            except (ValueError, IndexError):
                                continue
                    
                    # Create probability array
                    non_zero_count = 0
                    max_prob = 0.0
                    for linear_index in all_grid_cells:
                        prob = prob_by_index.get(linear_index, 0.0)
                        detection_probs.append(prob)
                        if prob > 0:
                            non_zero_count += 1
                        max_prob = max(max_prob, prob)
                    
                    # print(f"DEBUG: Detection probs for {detection_filename}: {len(detection_probs)} values, non-zero: {non_zero_count}, max: {max_prob:.3f}")
                
                # Store probability file
                detection_probs_formatted = "[\n" + ",\n".join([f"  {prob}" for prob in detection_probs]) + "\n]"
                probability_files[detection_filename] = detection_probs_formatted
            
            # Create location data
            location_data = {
                "coverage_metrics": coverage_metrics,
                "long": location["lng"],
                "lat": location["lat"]
            }
            
            main_json["sensors"][exact_sensor_name]["possible_locs"].append(location_data)
    
    main_json_string = json.dumps(main_json, indent=2)
    
    # print(f"DEBUG: Created {len(probability_files)} probability files")
    for filename in list(probability_files.keys())[:3]:  # Show first 3 filenames
        content = probability_files[filename]
        non_zero_count = len([x for x in content.split(',') if float(x.strip().replace('[', '').replace(']', '').replace('\n', '')) > 0])
        # print(f"DEBUG: File {filename} has {len(content.split(','))} total values, {non_zero_count} non-zero")
    
    return main_json_string, probability_files

def detection_probability_tab():
    """
    Preprocess Detection Probability Tab with support for multiple UAV specifications,
    generating both main JSON and separate probability files in a ZIP.
    Only creates detection probability files, but includes both target_detection and foe_or_friend sections that reference the same detection files.
    """
    st.header("Detection Probability Analysis")
    
    # Check if location has been selected first
    if not st.session_state.location_selected:
        st.warning("Please select a location in the initial screen first.")
        if st.button("Return to Location Selection"):
            st.session_state.location_selected = False
            st.rerun()
        return
    
    # Then check if an area of interest has been selected
    if not st.session_state.area_selected:
        st.warning("Please select an area of interest in the Map & Selection tab first.")
        return
    
    # Check if sensors are placed
    if not st.session_state.potential_locations:
        st.warning("Please place sensors in the Possible Sensor Placement tab first.")
        return
    
    # Check if UAV specifications are added
    if not st.session_state.get('uav_specifications_list'):
        st.warning("Please add at least one UAV configuration in the sidebar.")
        return
    
    # Display how many UAV configurations are available
    st.info(f"Found {len(st.session_state.uav_specifications_list)} UAV configuration(s) to analyze.")
    
    # Direct run button for detection probability calculation
    run_button = st.button("Run Detection Probability", type="primary")
    
    if run_button or st.session_state.get('detection_prob_calculated', False):
        if run_button:
            # Show a spinner while calculating
            with st.spinner("Calculating detection probabilities for all UAV configurations..."):
                # Check if we have a grid - UPDATED CONDITION
                if st.session_state.grid_enabled and "grid_size_degrees" in st.session_state:
                    # Create the grid data based on grid settings
                    sw_corner = st.session_state.sw_corner
                    ne_corner = st.session_state.ne_corner
                    grid_size_degrees = st.session_state.grid_size_degrees  # Changed variable name
                    
                    # Get center latitude for calculations
                    center_lat = (sw_corner[0] + ne_corner[0]) / 2
                    
                    # Grid size is already in degrees - calculate longitude adjustment
                    lat_grid_size = grid_size_degrees
                    lng_grid_size = grid_size_degrees / np.cos(np.radians(center_lat))
                    
                    # Calculate grid dimensions in degrees
                    width_degrees = ne_corner[1] - sw_corner[1]
                    height_degrees = ne_corner[0] - sw_corner[0]
                    
                    num_rows = int(np.ceil(height_degrees / lat_grid_size))
                    num_cols = int(np.ceil(width_degrees / lng_grid_size))
                    
                    # Create grid cells
                    grid_data = []
                    
                    for i in range(num_rows):
                        for j in range(num_cols):
                            # Calculate cell boundaries using degree-based grid
                            lat_sw = sw_corner[0] + i * lat_grid_size
                            lng_sw = sw_corner[1] + j * lng_grid_size
                            
                            lat_ne = lat_sw + lat_grid_size
                            lng_ne = lng_sw + lng_grid_size
                            
                            # Calculate cell center
                            center_lat_cell = (lat_sw + lat_ne) / 2
                            center_lng_cell = (lng_sw + lng_ne) / 2
                            
                            # Check if center is inside the boundary
                            if st.session_state.boundary_type == "rectangle":
                                # For rectangle, check if center is within the defined bounds
                                is_inside = (sw_corner[0] <= center_lat_cell <= ne_corner[0] and 
                                            sw_corner[1] <= center_lng_cell <= ne_corner[1])
                            elif st.session_state.boundary_type == "polygon":
                                is_inside = is_point_in_polygon(
                                    [center_lat_cell, center_lng_cell],
                                    st.session_state.boundary_points
                                )
                            else:
                                is_inside = True
                            # Only add cells inside the boundary
                            if is_inside:
                                grid_cell = {
                                    "grid_id": f"r{i}c{j}",
                                    "row": i,
                                    "col": j,
                                    "sw_corner": [lat_sw, lng_sw],
                                    "ne_corner": [lat_ne, lng_ne],
                                    "center": [center_lat_cell, center_lng_cell],
                                    "size_degrees": grid_size_degrees  # Changed: now stores degrees instead of km
                                }
                                grid_data.append(grid_cell)
                    
                    # Calculate detection probabilities for each UAV configuration
                    all_uav_results = {}
                    
                    for uav_spec in st.session_state.uav_specifications_list:
                        # Create UAV spec dictionary with required format
                        uav_specs = {
                            "altitude": uav_spec["altitude"],
                            "speed": uav_spec["speed"]
                        }
                        
                        # Calculate probabilities for this UAV
                        grid_probabilities = preprocess_detection_probability(
                            grid_data,
                            st.session_state.potential_locations,
                            st.session_state.sensor_specifications,
                            uav_specs
                        )
                        
                        # Add SW corner information to each grid cell in the results
                        for grid_id, prob_data in grid_probabilities.items():
                            # Find the corresponding grid cell from grid_data
                            for grid_cell in grid_data:
                                if grid_cell["grid_id"] == grid_id:
                                    prob_data["sw_corner"] = grid_cell["sw_corner"]
                                    break
                        
                        # Store results with UAV ID as key
                        all_uav_results[uav_spec["id"]] = {
                            "uav_spec": uav_spec,
                            "grid_probabilities": grid_probabilities
                        }
                    
                    # Store results in session state
                    st.session_state.all_uav_results = all_uav_results
                    st.session_state.detection_prob_calculated = True
                    
                    # Create the combined JSON output - main file and probability files
                    main_json, probability_files = create_combined_json_output(
                        all_uav_results,
                        st.session_state.potential_locations,
                        st.session_state.sensor_specifications,
                        st.session_state.uav_specifications_list,
                        st.session_state.boundary_type,
                        st.session_state.boundary_points,
                        st.session_state.sw_corner,
                        st.session_state.ne_corner,
                        grid_size_degrees  # Changed: now passing degrees instead of km
                    )
                    
                    # Store in session state
                    st.session_state.main_json = main_json
                    st.session_state.probability_files = probability_files
                    
                    # Create a zip file containing all JSON files
                    import zipfile
                    import io

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        # Add main JSON file
                        zipf.writestr("muscat_input_var_core.json", main_json)
                        
                        # Add all probability files (detection only)
                        for filename, content in probability_files.items():
                            zipf.writestr(filename, content)

                    zip_buffer.seek(0)
                    st.session_state.zip_data = zip_buffer.getvalue()

                    # Save the zip file to a persistent location for the analytical model
                    import os
                    from datetime import datetime

                    # Create output directory if it doesn't exist
                    output_dir = "detection_outputs"
                    os.makedirs(output_dir, exist_ok=True)

                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    zip_filename = f"muscat_detection_{timestamp}.zip"
                    zip_path = os.path.join(output_dir, zip_filename)

                    # Write the zip data to a file
                    with open(zip_path, 'wb') as f:
                        f.write(st.session_state.zip_data)

                    # IMPORTANT: Save the zip file path to session state
                    st.session_state.detection_zip_path = zip_path
                else:
                    st.error("Please enable and configure the grid in the Map & Selection tab first.")
                    return
        
        # Display results if they exist
        if st.session_state.get('detection_prob_calculated', False):
            # Add download button for the ZIP file
            st.success("Detection probability analysis completed!")
            
            st.download_button(
                label="Download All Detection Data (ZIP)",
                data=st.session_state.zip_data,
                file_name="muscat_input_var_core.zip",
                mime="application/zip",
                help="Download main JSON and detection probability files in a ZIP archive"
            )
            # Show that the file is saved and ready for the analytical model
            # if 'detection_zip_path' in st.session_state:
            #     st.info(f"✓ Detection data saved and ready for analytical model: {os.path.basename(st.session_state.detection_zip_path)}")