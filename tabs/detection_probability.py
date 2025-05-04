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

# Import from map_utils
from map_utils import is_point_in_rectangle, is_point_in_polygon

def calculate_detection_probability(sensor_type, distance, sensor_specs, uav_altitude, uav_speed):
    """
    Calculate the probability of detection based on sensor type, distance, and UAV parameters
    
    Parameters:
    -----------
    sensor_type : str
        Type of sensor (Radar, RF, LiDAR)
    distance : float
        Distance from sensor to the grid cell center in meters
    sensor_specs : dict
        Specifications of the sensor
    uav_altitude : float
        Altitude of the UAV in meters
    uav_speed : float
        Speed of the UAV in m/s
        
    Returns:
    --------
    float : Probability of detection (0.0 to 1.0)
    """
    # Get sensor detection range
    max_range = sensor_specs.get("detection_range", 1000)
    
    # Basic probability model - decreases with distance
    # If distance > max_range, probability is 0
    if distance > max_range:
        return 0.0
    
    # Base probability calculation
    # Using a gaussian-like falloff model where probability is highest at the center
    # and falls off with distance
    base_prob = math.exp(-(distance / max_range)**2)
    
    # Apply sensor type specific modifiers
    if sensor_type == "Radar":
        # Radar performance is affected by altitude and speed (doppler effect)
        # Higher altitudes might reduce effectiveness slightly
        altitude_factor = 1.0 - 0.1 * (uav_altitude / 1000)  # Small decrease with altitude
        
        # Faster moving targets can be easier to detect with radar
        speed_factor = 0.9 + 0.1 * min(uav_speed / 20, 1.0)  # Bonus for faster targets
        
        # RCS (Radar Cross Section) factor - simplified here
        rcs_factor = 0.95  # Assuming standard drone RCS
        
        final_prob = base_prob * altitude_factor * speed_factor * rcs_factor
    
    elif sensor_type == "RF":
        # RF detection depends on radio signals, less affected by altitude directly
        # but signal strength decreases with distance
        signal_factor = 1.0 - 0.3 * (distance / max_range)**1.5
        
        # RF signals can be affected by UAV speed indirectly
        speed_factor = 1.0  # Neutral effect for RF
        
        # RF signals can be affected by UAV altitude
        altitude_factor = 1.0 - 0.05 * (uav_altitude / 500)  # Slight decrease with altitude
        
        final_prob = base_prob * signal_factor * speed_factor * altitude_factor
    
    elif sensor_type == "LiDAR":
        # LiDAR heavily affected by distance and has shorter range
        range_factor = 1.0 - 0.5 * (distance / max_range)**1.2
        
        # LiDAR can be affected by UAV speed (faster = harder to get good returns)
        speed_factor = 1.0 - 0.2 * min(uav_speed / 30, 1.0)  # Penalty for faster targets
        
        # LiDAR effectiveness decreases with altitude
        altitude_factor = 1.0 - 0.2 * (uav_altitude / 300)  # Decrease with altitude
        
        final_prob = base_prob * range_factor * speed_factor * altitude_factor
    
    else:
        # Default case
        final_prob = base_prob
    
    # Add some randomness to simulate real-world variability
    # Use a normal distribution with mean=final_prob and sigma=0.05
    # But ensure result stays between 0 and 1
    randomness = max(0, min(1, np.random.normal(0, 0.05) + final_prob))
    
    # Ensure probability is between 0 and 1
    return max(0.0, min(1.0, randomness))

def preprocess_detection_probability(grid_data, sensor_locations, sensor_specifications, uav_specs):
    """
    Preprocess and calculate detection probabilities for each grid cell
    
    Parameters:
    -----------
    grid_data : list
        List of grid cells with coordinates
    sensor_locations : list
        List of sensor locations with coordinates and names
    sensor_specifications : list
        List of sensor specifications
    uav_specs : dict
        UAV altitude and speed
        
    Returns:
    --------
    dict : Grid probabilities with detection probabilities per cell and sensor
    """
    results = {}
    
    # Ensure we have both sensor locations and specs
    if not sensor_locations or not sensor_specifications:
        return results
    
    # Extract UAV parameters
    uav_altitude = uav_specs.get("altitude", 100)
    uav_speed = uav_specs.get("speed", 10)
    
    # Process each grid cell
    for grid_cell in grid_data:
        grid_id = grid_cell["grid_id"]
        cell_center = grid_cell["center"]  # [lat, lng]
        
        # Initialize probabilities for this cell
        cell_probs = {
            "grid_id": grid_id,
            "center": cell_center,
            "sensors": {},
            "combined_probability": 0.0
        }
        
        # Calculate probabilities for each sensor
        for i, sensor in enumerate(sensor_locations):
            # Get sensor position
            sensor_pos = [sensor["lat"], sensor["lng"]]
            sensor_name = sensor.get("name", f"Sensor {i+1}")
            
            # Check if we have specifications for this sensor
            if i < len(sensor_specifications):
                sensor_spec = sensor_specifications[i]
                sensor_type = sensor_spec.get("type", "Unknown")
                
                # Calculate distance between sensor and grid cell center
                # Convert lat/lng to approximate meters (very simplified)
                # 1 degree lat ≈ 111,000 meters, 1 degree lng ≈ 111,000 * cos(lat) meters
                lat1, lng1 = sensor_pos
                lat2, lng2 = cell_center
                
                # Convert to radians
                lat1_rad = math.radians(lat1)
                lng1_rad = math.radians(lng1)
                lat2_rad = math.radians(lat2)
                lng2_rad = math.radians(lng2)
                
                # Haversine formula for distance
                earth_radius = 6371000  # Earth radius in meters
                dlat = lat2_rad - lat1_rad
                dlng = lng2_rad - lng1_rad
                a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                distance = earth_radius * c  # Distance in meters
                
                # Calculate probability for this sensor
                prob = calculate_detection_probability(
                    sensor_type,
                    distance,
                    sensor_spec,
                    uav_altitude,
                    uav_speed
                )
                
                # Store probability
                cell_probs["sensors"][sensor_name] = {
                    "probability": prob,
                    "distance": distance,
                    "type": sensor_type
                }
            else:
                # Default if no specification
                cell_probs["sensors"][sensor_name] = {
                    "probability": 0.0,
                    "distance": 0.0,
                    "type": "Unknown"
                }
        
        # Calculate combined probability using the formula:
        # P(detection) = 1 - P(all sensors fail to detect)
        # P(all fail) = (1-P1) * (1-P2) * ... * (1-Pn)
        if cell_probs["sensors"]:
            p_all_fail = 1.0
            for sensor_name, sensor_data in cell_probs["sensors"].items():
                p_all_fail *= (1.0 - sensor_data["probability"])
            
            combined_prob = 1.0 - p_all_fail
            cell_probs["combined_probability"] = combined_prob
        
        # Store the results
        results[grid_id] = cell_probs
    
    return results

def generate_probability_heatmap(grid_probabilities, boundary_type, boundary_points=None, sw_corner=None, ne_corner=None):
    """
    Generate a heatmap visualization of detection probabilities
    
    Parameters:
    -----------
    grid_probabilities : dict
        Dictionary of grid probabilities
    boundary_type : str
        Type of boundary (rectangle or polygon)
    boundary_points : list, optional
        List of boundary points for polygon
    sw_corner : list, optional
        Southwest corner for rectangle
    ne_corner : list, optional
        Northeast corner for rectangle
        
    Returns:
    --------
    bytes : PNG image data as bytes
    """
    # Extract grid data for visualization
    grid_ids = []
    centers = []
    probs = []
    
    for grid_id, data in grid_probabilities.items():
        grid_ids.append(grid_id)
        centers.append(data["center"])
        probs.append(data["combined_probability"])
    
    # Convert to numpy arrays
    centers = np.array(centers)
    probs = np.array(probs)
    
    # Create a figure
    fig, ax = plt.figure(figsize=(10, 8), dpi=100), plt.gca()
    
    # Create custom colormap (red to green)
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]  # R -> Y -> G
    cmap = LinearSegmentedColormap.from_list("RYG", colors, N=256)
    
    # Plot the boundary
    if boundary_type == "rectangle" and sw_corner and ne_corner:
        # Create a rectangle patch
        min_lat, min_lng = sw_corner
        max_lat, max_lng = ne_corner
        ax.plot([min_lng, max_lng, max_lng, min_lng, min_lng], 
                [min_lat, min_lat, max_lat, max_lat, min_lat], 
                'k-', linewidth=2)
    elif boundary_type == "polygon" and boundary_points:
        # Extract lat/lng points
        lats = [p[0] for p in boundary_points]
        lngs = [p[1] for p in boundary_points]
        # Close the polygon
        lats.append(lats[0])
        lngs.append(lngs[0])
        ax.plot(lngs, lats, 'k-', linewidth=2)
    
    # Create a scatter plot with probability-based colors
    scatter = ax.scatter(centers[:, 1], centers[:, 0], c=probs, cmap=cmap, 
                        s=100, alpha=0.7, edgecolors='k', linewidths=0.5)
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Detection Probability')
    
    # Set plot limits based on the boundary
    if boundary_type == "rectangle" and sw_corner and ne_corner:
        min_lat, min_lng = sw_corner
        max_lat, max_lng = ne_corner
        # Add some padding
        padding = 0.01  # degrees
        ax.set_xlim(min_lng - padding, max_lng + padding)
        ax.set_ylim(min_lat - padding, max_lat + padding)
    elif boundary_type == "polygon" and boundary_points:
        lats = [p[0] for p in boundary_points]
        lngs = [p[1] for p in boundary_points]
        # Add some padding
        padding = 0.01  # degrees
        ax.set_xlim(min(lngs) - padding, max(lngs) + padding)
        ax.set_ylim(min(lats) - padding, max(lats) + padding)
    
    # Set labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('UAV Detection Probability Heatmap')
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Close the figure to free memory
    plt.close(fig)
    
    # Return the image data
    return buf

def detection_probability_tab():
    """
    Preprocess Detection Probability Tab
    """
    st.header("Detection Probability Preprocessing")
    
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
    
    # Create layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Detection Probability Parameters")
        
        # Show sensor locations and types
        st.write("### Sensor Locations")
        
        # Display sensors in a table
        if st.session_state.potential_locations:
            sensor_data = []
            for i, location in enumerate(st.session_state.potential_locations):
                # Get sensor specifications if available
                sensor_spec = None
                sensor_type = "Unknown"
                detection_range = "N/A"
                
                if i < len(st.session_state.sensor_specifications):
                    sensor_spec = st.session_state.sensor_specifications[i]
                    sensor_type = sensor_spec.get("type", "Unknown")
                    detection_range = f"{sensor_spec.get('detection_range', 'N/A')} m"
                
                sensor_data.append({
                    "Name": location.get("name", f"Sensor {i+1}"),
                    "Latitude": f"{location['lat']:.6f}",
                    "Longitude": f"{location['lng']:.6f}",
                    "Type": sensor_type,
                    "Range": detection_range
                })
            
            # Convert to DataFrame for better display
            sensors_df = pd.DataFrame(sensor_data)
            st.dataframe(sensors_df, use_container_width=True)
        
        # Show UAV settings
        st.write("### UAV Settings")
        
        # Get UAV settings from session state
        uav_altitude = st.session_state.get("uav_altitude", 100.0)
        uav_speed = st.session_state.get("uav_speed", 10.0)
        
        # Create a form for UAV settings
        with st.form("uav_settings_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                uav_altitude = st.number_input(
                    "UAV Altitude (meters)", 
                    min_value=10.0,
                    max_value=5000.0,
                    value=uav_altitude,
                    step=10.0
                )
            
            with col_b:
                uav_speed = st.number_input(
                    "UAV Speed (m/s)",
                    min_value=1.0,
                    max_value=100.0,
                    value=uav_speed,
                    step=1.0
                )
            
            # Store UAV settings in session state
            if st.form_submit_button("Update UAV Settings"):
                st.session_state.uav_altitude = uav_altitude
                st.session_state.uav_speed = uav_speed
                st.success("UAV settings updated!")
        
        # Add advanced settings expander
        with st.expander("Advanced Settings"):
            # Random seed for reproducible results
            random_seed = st.number_input(
                "Random Seed", 
                min_value=0,
                value=42,
                help="Set a random seed for reproducible results"
            )
            
            # Set the random seed
            np.random.seed(random_seed)
            
            # Add options for probability calculation models
            prob_model = st.selectbox(
                "Probability Model",
                ["Gaussian Falloff", "Linear Falloff", "Exponential Falloff"],
                index=0,
                help="Select the mathematical model for calculating detection probability based on distance"
            )
            
            # Add options for combining probabilities
            combination_method = st.selectbox(
                "Probability Combination Method",
                ["Independent Sensors (Default)", "Maximum Probability", "Weighted Average"],
                index=0,
                help="Method for combining probabilities from multiple sensors"
            )
        
        # Button to run detection probability calculation
        run_button = st.button("Run Detection Probability Analysis", type="primary")
        
        # Store calculation status in session state
        if "detection_prob_calculated" not in st.session_state:
            st.session_state.detection_prob_calculated = False
        
        if run_button:
            # Show a spinner while calculating
            with st.spinner("Calculating detection probabilities..."):
                # Check if we have a grid
                if st.session_state.grid_enabled and "grid_size_km" in st.session_state:
                    # Create the grid data based on grid settings
                    # This would normally be handled by the map_selection_tab function
                    # Here we'll recreate it for our calculations
                    sw_corner = st.session_state.sw_corner
                    ne_corner = st.session_state.ne_corner
                    grid_size_km = st.session_state.grid_size_km
                    
                    # Get center latitude for calculations
                    center_lat = (sw_corner[0] + ne_corner[0]) / 2
                    
                    # Convert grid size from km to degrees
                    # 1 degree of latitude is approximately 111 km
                    lat_deg_per_km = 1 / 111.0
                    grid_size_lat = grid_size_km * lat_deg_per_km
                    
                    # 1 degree of longitude varies with latitude
                    lng_deg_per_km = 1 / (111.0 * np.cos(np.radians(center_lat)))
                    grid_size_lng = grid_size_km * lng_deg_per_km
                    
                    # Calculate grid dimensions
                    width_km = (ne_corner[1] - sw_corner[1]) / lng_deg_per_km
                    height_km = (ne_corner[0] - sw_corner[0]) / lat_deg_per_km
                    
                    num_rows = int(np.ceil(height_km / grid_size_km))
                    num_cols = int(np.ceil(width_km / grid_size_km))
                    
                    # Create grid cells
                    grid_data = []
                    
                    for i in range(num_rows):
                        for j in range(num_cols):
                            # Calculate cell boundaries
                            lat_sw = sw_corner[0] + i * grid_size_lat
                            lng_sw = sw_corner[1] + j * grid_size_lng
                            
                            lat_ne = lat_sw + grid_size_lat
                            lng_ne = lng_sw + grid_size_lng
                            
                            # Calculate cell center
                            center_lat = (lat_sw + lat_ne) / 2
                            center_lng = (lng_sw + lng_ne) / 2
                            
                            # Check if center is inside the boundary
                            is_inside = True
                            if st.session_state.boundary_type == "polygon":
                                is_inside = is_point_in_polygon(
                                    [center_lat, center_lng],
                                    st.session_state.boundary_points
                                )
                            
                            # Only add cells inside the boundary
                            if is_inside:
                                grid_cell = {
                                    "grid_id": f"r{i}c{j}",
                                    "row": i,
                                    "col": j,
                                    "sw_corner": [lat_sw, lng_sw],
                                    "ne_corner": [lat_ne, lng_ne],
                                    "center": [center_lat, center_lng],
                                    "size_km": grid_size_km
                                }
                                grid_data.append(grid_cell)
                    
                    # Calculate detection probabilities
                    uav_specs = {
                        "altitude": uav_altitude,
                        "speed": uav_speed
                    }
                    
                    grid_probabilities = preprocess_detection_probability(
                        grid_data,
                        st.session_state.potential_locations,
                        st.session_state.sensor_specifications,
                        uav_specs
                    )
                    
                    # Store results in session state
                    st.session_state.grid_probabilities = grid_probabilities
                    st.session_state.detection_prob_calculated = True
                    
                    # Generate heatmap
                    heatmap_buf = generate_probability_heatmap(
                        grid_probabilities,
                        st.session_state.boundary_type,
                        st.session_state.boundary_points,
                        st.session_state.sw_corner,
                        st.session_state.ne_corner
                    )
                    
                    # Store the heatmap in session state
                    st.session_state.prob_heatmap = heatmap_buf
                    
                    st.success("Detection probability analysis completed!")
                else:
                    st.error("Please enable and configure the grid in the Map & Selection tab first.")
    
    with col2:
        st.subheader("Results Visualization")
        
        # Show the detection probability heatmap if calculated
        if st.session_state.detection_prob_calculated and "prob_heatmap" in st.session_state:
            st.image(st.session_state.prob_heatmap, caption="Detection Probability Heatmap")
            
            # Add a download button for the heatmap
            heatmap_bytes = st.session_state.prob_heatmap.getvalue()
            b64 = base64.b64encode(heatmap_bytes).decode()
            href = f'<a href="data:image/png;base64,{b64}" download="detection_probability_heatmap.png">Download Heatmap</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Display statistics about the results
            if "grid_probabilities" in st.session_state:
                grid_probs = st.session_state.grid_probabilities
                
                if grid_probs:
                    # Calculate statistics
                    probs = [data["combined_probability"] for data in grid_probs.values()]
                    avg_prob = np.mean(probs)
                    min_prob = np.min(probs)
                    max_prob = np.max(probs)
                    median_prob = np.median(probs)
                    
                    # Show statistics
                    st.write("### Detection Statistics")
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric("Average Probability", f"{avg_prob:.2%}")
                        st.metric("Minimum Probability", f"{min_prob:.2%}")
                    
                    with col_b:
                        st.metric("Maximum Probability", f"{max_prob:.2%}")
                        st.metric("Median Probability", f"{median_prob:.2%}")
                    
                    # Show coverage statistics
                    low_coverage = len([p for p in probs if p < 0.3])
                    med_coverage = len([p for p in probs if 0.3 <= p < 0.7])
                    high_coverage = len([p for p in probs if p >= 0.7])
                    total_cells = len(probs)
                    
                    st.write("### Coverage Analysis")
                    st.write(f"Total grid cells: {total_cells}")
                    
                    # Create a DataFrame for the coverage stats
                    coverage_data = {
                        "Coverage Level": ["Low (0-30%)", "Medium (30-70%)", "High (70-100%)"],
                        "Cell Count": [low_coverage, med_coverage, high_coverage],
                        "Percentage": [
                            f"{low_coverage/total_cells:.1%}",
                            f"{med_coverage/total_cells:.1%}",
                            f"{high_coverage/total_cells:.1%}"
                        ]
                    }
                    coverage_df = pd.DataFrame(coverage_data)
                    st.dataframe(coverage_df, use_container_width=True)
                    
                    # Create a pie chart of coverage levels
                    fig, ax = plt.figure(figsize=(6, 6)), plt.gca()
                    ax.pie([low_coverage, med_coverage, high_coverage],
                          labels=["Low", "Medium", "High"],
                          autopct='%1.1f%%',
                          colors=['#ff6666', '#ffcc66', '#66cc66'],
                          startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)
                    
                    # Export results option
                    st.write("### Export Results")
                    
                    # Convert grid probabilities to CSV
                    csv_data = []
                    for grid_id, data in grid_probs.items():
                        row = {
                            "Grid ID": grid_id,
                            "Latitude": data["center"][0],
                            "Longitude": data["center"][1],
                            "Combined Probability": data["combined_probability"]
                        }
                        
                        # Add individual sensor probabilities
                        for sensor_name, sensor_data in data["sensors"].items():
                            row[f"{sensor_name} Probability"] = sensor_data["probability"]
                            row[f"{sensor_name} Distance (m)"] = sensor_data["distance"]
                        
                        csv_data.append(row)
                    
                    # Convert to DataFrame and then to CSV
                    results_df = pd.DataFrame(csv_data)
                    csv = results_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name='detection_probability_results.csv',
                        mime='text/csv',
                    )
                    
                    # Export as GeoJSON option
                    geojson_data = {
                        "type": "FeatureCollection",
                        "features": []
                    }
                    
                    for grid_id, data in grid_probs.items():
                        # Create a feature for this grid cell
                        feature = {
                            "type": "Feature",
                            "properties": {
                                "grid_id": grid_id,
                                "combined_probability": data["combined_probability"]
                            },
                            "geometry": {
                                "type": "Point",
                                "coordinates": [data["center"][1], data["center"][0]]
                            }
                        }
                        
                        # Add individual sensor probabilities to properties
                        for sensor_name, sensor_data in data["sensors"].items():
                            feature["properties"][f"{sensor_name}_probability"] = sensor_data["probability"]
                            feature["properties"][f"{sensor_name}_distance"] = sensor_data["distance"]
                        
                        geojson_data["features"].append(feature)
                    
                    # Convert to JSON string
                    geojson_str = json.dumps(geojson_data, indent=2)
                    
                    st.download_button(
                        label="Download Results as GeoJSON",
                        data=geojson_str,
                        file_name='detection_probability_results.geojson',
                        mime='application/geo+json',
                    )
        else:
            st.info("Run the detection probability analysis to see results here.")
            
            # Show a placeholder image
            st.image("https://via.placeholder.com/600x400.png?text=Detection+Probability+Heatmap", 
                    caption="Placeholder for detection probability heatmap")