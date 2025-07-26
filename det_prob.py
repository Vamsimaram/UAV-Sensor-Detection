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
    # Get sensor detection range in km
    max_range_km = sensor_specs.get("detection_range", 10.0)
    
    # Basic probability model - decreases with distance
    # If distance > max_range, probability is 0
    if distance_km > max_range_km:
        return 0.0
    
    # Base probability calculation
    # Using a gaussian-like falloff model where probability is highest at the center
    # and falls off with distance
    base_prob = math.exp(-(distance_km / max_range_km)**2)
    
    # Apply sensor type specific modifiers
    if sensor_type.lower() == "radar":
        # Radar performance is affected by altitude and speed (doppler effect)
        # Higher altitudes might reduce effectiveness slightly
        altitude_factor = 1.0 - 0.1 * (uav_altitude_km / 1.0)  # Small decrease with altitude (1km reference)
        altitude_factor = max(0.1, altitude_factor)  # Don't go below 0.1
        
        # Faster moving targets can be easier to detect with radar
        speed_factor = 0.9 + 0.1 * min(uav_speed_kmh / 72, 1.0)  # Bonus for faster targets (72 km/h reference)
        
        # RCS (Radar Cross Section) factor - simplified here
        rcs_factor = 0.95  # Assuming standard drone RCS
        
        final_prob = base_prob * altitude_factor * speed_factor * rcs_factor
    
    elif sensor_type.lower() == "rf":
        # RF detection depends on radio signals, less affected by altitude directly
        # but signal strength decreases with distance
        signal_factor = 1.0 - 0.3 * (distance_km / max_range_km)**1.5
        signal_factor = max(0.1, signal_factor)  # Don't go below 0.1
        
        # RF signals can be affected by UAV speed indirectly
        speed_factor = 1.0  # Neutral effect for RF
        
        # RF signals can be affected by UAV altitude
        altitude_factor = 1.0 - 0.05 * (uav_altitude_km / 0.5)  # Slight decrease with altitude (0.5km reference)
        altitude_factor = max(0.1, altitude_factor)  # Don't go below 0.1
        
        final_prob = base_prob * signal_factor * speed_factor * altitude_factor
    
    elif sensor_type.lower() == "lidar":
        # LiDAR heavily affected by distance and has shorter range
        range_factor = 1.0 - 0.5 * (distance_km / max_range_km)**1.2
        range_factor = max(0.1, range_factor)  # Don't go below 0.1
        
        # LiDAR can be affected by UAV speed (faster = harder to get good returns)
        speed_factor = 1.0 - 0.2 * min(uav_speed_kmh / 108, 1.0)  # Penalty for faster targets (108 km/h reference)
        speed_factor = max(0.1, speed_factor)  # Don't go below 0.1
        
        # LiDAR effectiveness decreases with altitude
        altitude_factor = 1.0 - 0.2 * (uav_altitude_km / 0.3)  # Decrease with altitude (0.3km reference)
        altitude_factor = max(0.1, altitude_factor)  # Don't go below 0.1
        
        final_prob = base_prob * range_factor * speed_factor * altitude_factor
    
    else:
        # Default case
        final_prob = base_prob
    
    # Add some randomness to simulate real-world variability
    # Use a normal distribution with mean=final_prob and sigma=0.05
    # But ensure result stays between 0 and 1
    randomness = max(0, min(1, np.random.normal(0, 0.05) + final_prob))
    
    # Ensure probability is between 0 and 1 and round to 3 decimal places
    return round(max(0.0, min(1.0, randomness)), 3)

def preprocess_detection_probability(grid_data, sensor_locations, sensor_specifications, uav_specs):
    """
    Preprocess and calculate detection probabilities for each grid cell, 
    organized by grid cell, sensor location, and sensor type
    
    Parameters:
    -----------
    grid_data : list
        List of grid cells with coordinates
    sensor_locations : list
        List of sensor locations with coordinates and names
    sensor_specifications : list
        List of sensor specifications
    uav_specs : dict
        UAV altitude (km) and speed (km/h)
        
    Returns:
    --------
    dict : Grid probabilities with detection probabilities per cell, sensor location, and sensor type
    """
    results = {}
    
    # Ensure we have both sensor locations and specs
    if not sensor_locations or not sensor_specifications:
        return results
    
    # Extract UAV parameters (now in km and km/h)
    uav_altitude_km = uav_specs.get("altitude", 0.1)  # Default 0.1 km (100m)
    uav_speed_kmh = uav_specs.get("speed", 36)  # Default 36 km/h (10 m/s)
    
    # Process each grid cell
    for grid_cell in grid_data:
        grid_id = grid_cell["grid_id"]
        cell_center = grid_cell["center"]  # [lat, lng]
        
        # Initialize data structure for this cell
        cell_data = {
            "grid_id": grid_id,
            "center": cell_center,
            "sensor_locations": {}
        }
        
        # Calculate probabilities for each sensor location
        for i, sensor in enumerate(sensor_locations):
            # Get sensor position
            sensor_pos = [sensor["lat"], sensor["lng"]]
            sensor_name = sensor.get("name", f"Sensor_{i+1}")
            
            # Initialize data structure for this sensor location
            sensor_location_data = {
                "position": sensor_pos,
                "sensor_types": {}
            }
            
            # Calculate distance between sensor and grid cell center using Haversine formula
            lat1, lng1 = sensor_pos
            lat2, lng2 = cell_center
            
            # Convert to radians
            lat1_rad = math.radians(lat1)
            lng1_rad = math.radians(lng1)
            lat2_rad = math.radians(lat2)
            lng2_rad = math.radians(lng2)
            
            # Haversine formula for distance
            earth_radius_km = 6371.0  # Earth radius in kilometers
            dlat = lat2_rad - lat1_rad
            dlng = lng2_rad - lng1_rad
            a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance_km = earth_radius_km * c  # Distance in kilometers
            
            # Store distance at sensor location level
            sensor_location_data["distance"] = distance_km
            
            # Calculate probabilities for each sensor model at this location
            for j, sensor_spec in enumerate(sensor_specifications):
                sensor_type = sensor_spec.get("type", "Unknown")
                model = sensor_spec.get("model", "Unknown")
                
                # Calculate probability for this sensor model at this location
                prob = calculate_detection_probability(
                    sensor_type,
                    distance_km,
                    sensor_spec,
                    uav_altitude_km,
                    uav_speed_kmh
                )
                
                # Store sensor type data using the type as key
                # Simplified to only include probability and model
                sensor_location_data["sensor_types"][sensor_type] = {
                    "probability": prob,
                    "model": model
                }
            
            # Store the sensor location data
            cell_data["sensor_locations"][sensor_name] = sensor_location_data
        
        # Store the results
        results[grid_id] = cell_data
    
    return results

def create_combined_json_output(all_uav_results, sensor_locations, sensor_specifications, 
                            uav_specifications_list, boundary_type, boundary_points, 
                            sw_corner, ne_corner, grid_size_degrees):  # Changed parameter name
    """
    Create a combined output with main JSON file and separate probability files.
    All distance units are in meters for the final JSON output.
    Only creates detection probability files, but includes both target_detection and foe_or_friend sections that reference the same detection files.
    """
    import json
    import numpy as np
    
    # Calculate center latitude for calculations
    if sw_corner and ne_corner:
        center_lat = (sw_corner[0] + ne_corner[0]) / 2
    else:
        lats = [point[0] for point in boundary_points]
        center_lat = sum(lats) / len(lats)
    
    # Grid size is already in degrees - no conversion needed
    lat_grid_size = grid_size_degrees
    lng_grid_size = grid_size_degrees / np.cos(np.radians(center_lat))
    
    # Calculate the width and height of the area in degrees
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
    
    # Create target_types dictionary from UAV specifications
    target_types = {}
    for uav_spec in uav_specifications_list:
        uav_name = uav_spec.get("name", uav_spec.get("id", "UAV"))
        # Convert units: altitude from km to m, speed from km/h to m/s
        altitude_m = uav_spec.get("altitude", 0.1) * 1000  # Convert km to m
        speed_ms = uav_spec.get("speed", 36) / 3.6  # Convert km/h to m/s
        
        target_types[uav_name] = {
            "min_altitude": altitude_m,
            "max_velocity": speed_ms
        }
    
    # Create binary classifications for both target_detection and foe_or_friend
    binary_classifications = {}
    for uav_spec in uav_specifications_list:
        uav_name = uav_spec.get("name", uav_spec.get("id", "UAV"))
        binary_classifications[uav_name] = {"Pd_TH": 0.8}
    
    # Create the main JSON structure
    main_json = {
        "sensor_types": [],
        "sensors": {},
        "coverage_areas": {},  # This will be populated with protected areas
        "title": "Sensor Detection Analysis",
        "config": {
            "y_length": num_rows,
            "long_lat_unit": "degree",
            "x_length": num_cols,
            "sq_side_units": "long_lat_degree",
            "square_side": grid_size_degrees,  # Changed: now directly using degrees
            "time_unit": "hour",
            "earth_radius": 6371000,  # In meters
            "binary_classifications": {
                "target_detection": binary_classifications,
                "foe_or_friend": binary_classifications  # Use same classifications for both
            },
            "distance_unit": "m",  # Using meters
            "south_west_corner": {
                "long": sw_corner[1] if sw_corner else min([p[1] for p in boundary_points]),
                "lat": sw_corner[0] if sw_corner else min([p[0] for p in boundary_points])
            },
            "target_types": target_types
        },
        "area_of_interest": []
    }
    # Dictionary to store all probability files
    probability_files = {}
    
    # Initialize grid_cells_info here FIRST
    grid_cells_info = {}
    
    # Get all grid cells and their centers for protected area calculations
    if all_uav_results:
        first_uav_id = list(all_uav_results.keys())[0]
        grid_probabilities = all_uav_results[first_uav_id]["grid_probabilities"]
        
        # Extract all grid cell IDs, centers, and SW coordinates
        for grid_id, data in grid_probabilities.items():
            # Parse grid_id format "r{row}c{col}" to get linear index
            if grid_id.startswith('r') and 'c' in grid_id:
                try:
                    parts = grid_id[1:].split('c')  # Remove 'r' and split by 'c'
                    row = int(parts[0])
                    col = int(parts[1])
                    # Convert 2D grid coordinates to linear index
                    linear_index = row * num_cols + col
                    grid_cells_info[linear_index] = {
                        'grid_id': grid_id,
                        'center': data['center'],  # [lat, lng] - still needed for protected area calculations
                        'sw_corner': data.get('sw_corner', data['center'])  # [lat, lng] - SW corner coordinates
                    }
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse grid_id {grid_id}")
                    continue
    
    # Add area of interest - all SW corner points of grid cells inside the area of interest
    area_of_interest_sw_corners = []
    
    if grid_cells_info:
        for linear_index, cell_info in grid_cells_info.items():
            cell_center = cell_info['center']  # [lat, lng] - use center for area checking
            cell_sw_corner = cell_info['sw_corner']  # [lat, lng] - use SW corner for output
            
            # Add SW corner to area of interest (all grid cells are already filtered to be inside the area)
            area_of_interest_sw_corners.append(cell_sw_corner)
    
    main_json["area_of_interest"] = area_of_interest_sw_corners
    
    # Extract unique sensor types from sensor specifications
    sensor_types = list(set(spec["type"].lower() for spec in sensor_specifications))
    main_json["sensor_types"] = sensor_types
    
    # Process protected areas to determine coverage areas
    # Import the is_point_in_polygon function from map_utils
    from map_utils import is_point_in_polygon
    
    # Get protected areas from session state (this would need to be passed as a parameter)
    # For now, we'll check if it exists in st.session_state
    import streamlit as st
    protected_areas = st.session_state.get('protected_areas', [])
    
    if protected_areas:
        # Process each protected area
        for area in protected_areas:
            area_name = area['name']
            area_points = area['points']  # List of [lat, lng] points
            
            # Find all grid cells that fall within this protected area
            cells_in_area = []
            
            for linear_index, cell_info in grid_cells_info.items():
                cell_center = cell_info['center']  # [lat, lng] - use center for area checking
                
                # Check if the cell center is inside the protected area polygon
                if is_point_in_polygon(cell_center, area_points):
                    cells_in_area.append(linear_index)
            
            # Sort the cells for consistent ordering
            cells_in_area.sort()
            
            # Add to coverage areas if there are cells in this area
            if cells_in_area:
                main_json["coverage_areas"][area_name] = {
                    "area": cells_in_area
                }
    
    # If no protected areas or no cells in protected areas, create a default coverage area
    if not main_json["coverage_areas"]:
        # Use all grid cells as a fallback
        all_grid_cells = sorted(list(grid_cells_info.keys()))
        main_json["coverage_areas"]["default_area"] = {
            "area": all_grid_cells
        }
    
    # Group sensor specifications by type to avoid duplicates
    sensors_by_type = {}
    for i, sensor_spec in enumerate(sensor_specifications):
        sensor_type = sensor_spec["type"].lower()
        if sensor_type not in sensors_by_type:
            sensors_by_type[sensor_type] = []
        sensors_by_type[sensor_type].append((i, sensor_spec))
    
    # Process each unique sensor type
    for sensor_type, type_sensors in sensors_by_type.items():
        for type_idx, (spec_idx, sensor_spec) in enumerate(type_sensors):
            sensor_model = sensor_spec["model"]
            sensor_id = f"{sensor_type}{type_idx+1}"
            
            # Create sensor entry
            main_json["sensors"][sensor_id] = {
                "possible_locs": [],
                "optional_params": "Sensor parameters for detection model",
                "ppu": sensor_spec.get("price_per_unit", 500000),
                "model": sensor_model,
                "detection_period": 0.01,
                "type": sensor_type,
                "make": sensor_spec.get("manufacturer", "make_" + sensor_type)
            }
            
            # Process each possible location for this sensor
            for j, location in enumerate(sensor_locations):
                # Create coverage_metrics dictionary for each UAV type
                coverage_metrics = {}
                
                # Process each UAV specification to create separate entries
                for uav_spec in uav_specifications_list:
                    uav_name = uav_spec.get("name", uav_spec.get("id", "UAV"))
                    uav_id = uav_spec.get("id", uav_name)
                    
                    # Create file name for detection only
                    detection_filename = f"{sensor_id}_loc{j+1}_{uav_name}_detection.json"
                    
                    # Add UAV-specific coverage metrics (both target_detection and foe_or_friend)
                    coverage_metrics[uav_name] = {
                        "target_detection": {
                            "Pd": {"@file": detection_filename},
                            "Pfa": "tbd"
                        },
                        "foe_or_friend": {
                            "Pd": {"@file": detection_filename}  # Use same detection file
                        }
                    }
                    
                    # Create detection probabilities data for this location and UAV
                    detection_probs = []
                    
                    # Get all grid cells (sorted by linear index) for consistent ordering
                    all_grid_cells = sorted(list(grid_cells_info.keys()))
                    
                    # Get probabilities for this specific UAV configuration
                    if uav_id in all_uav_results:
                        grid_probabilities = all_uav_results[uav_id]["grid_probabilities"]
                        
                        # Create a dictionary to store probabilities by linear index
                        prob_by_index = {}
                        
                        # Extract probabilities for each grid cell
                        for grid_id, data in grid_probabilities.items():
                            # Parse grid_id to get linear index
                            if grid_id.startswith('r') and 'c' in grid_id:
                                try:
                                    parts = grid_id[1:].split('c')
                                    row = int(parts[0])
                                    col = int(parts[1])
                                    linear_index = row * num_cols + col
                                    
                                    # Find the probability for this sensor type at this location
                                    prob = 0.0
                                    sensor_name = location.get("name", f"Sensor_{j+1}")
                                    
                                    if sensor_name in data["sensor_locations"]:
                                        sensor_data = data["sensor_locations"][sensor_name]
                                        
                                        # Find the probability for this specific sensor model
                                        for s_type, type_data in sensor_data["sensor_types"].items():
                                            if (s_type.lower() == sensor_type.lower() and 
                                                type_data["model"] == sensor_model):
                                                prob = type_data["probability"]
                                                break
                                    
                                    # Store probability by linear index
                                    prob_by_index[linear_index] = prob
                                except (ValueError, IndexError):
                                    continue
                        
                        # Create probability arrays in the same order as all grid cells
                        for linear_index in all_grid_cells:
                            prob = prob_by_index.get(linear_index, 0.0)
                            detection_probs.append(prob)
                    
                    # Create the JSON file for detection only
                    detection_probs_formatted = "[\n" + ",\n".join([f"  {prob}" for prob in detection_probs]) + "\n]"
                    probability_files[detection_filename] = detection_probs_formatted
                
                # Create location data with coverage metrics for all UAVs
                location_data = {
                    "coverage_metrics": coverage_metrics,
                    "long": location["lng"],  # Correct longitude assignment
                    "lat": location["lat"]    # Correct latitude assignment
                }
                
                # Add location to sensor's possible locations
                main_json["sensors"][sensor_id]["possible_locs"].append(location_data)
    
    # Convert main JSON to string with indentation
    main_json_string = json.dumps(main_json, indent=2)
    
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
    run_button = st.button("Run Detection Probability Analysis & Generate JSON", type="primary")
    
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