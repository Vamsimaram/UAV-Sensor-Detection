import streamlit as st
import folium
from streamlit_folium import folium_static, st_folium
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
from pymongo import MongoClient
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# Import functions from other modules
from map_utils import create_grid_overlay, create_boundary_drawing_map, is_point_in_polygon, is_point_in_rectangle
from tabs.map_selection import map_selection_tab
from tabs.sensor import sensor_tab
from tabs.protected_areas import protected_areas_tab
from tabs.detection_probability import detection_probability_tab

# Set up the Streamlit page configuration
st.set_page_config(page_title="UAV Sensor Detection Probability Calculator", 
                   layout="wide", 
                   initial_sidebar_state="expanded")

def geocode_location(location_name):
    """
    Convert a location name to latitude/longitude coordinates using Nominatim geocoder
    
    Parameters:
    -----------
    location_name : str
        The name of the location to geocode
        
    Returns:
    --------
    tuple or None
        (latitude, longitude) of the location if successful, None otherwise
    """
    try:
        # Initialize the geocoder with a custom user agent
        geolocator = Nominatim(user_agent="uav_sensor_detection_app")
        
        # Geocode the location
        location = geolocator.geocode(location_name)
        
        if location:
            return (location.latitude, location.longitude)
        else:
            return None
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        st.error(f"Geocoding error: {str(e)}")
        return None

def load_sensor_data():
    """
    Load sensor specifications from the sensor-data.json file
    
    Returns:
    --------
    dict
        Dictionary containing sensor data
    """
    try:
        with open("sensor-data.json", "r") as f:
            sensor_data = json.load(f)
        return sensor_data
    except Exception as e:
        st.error(f"Error loading sensor data: {str(e)}")
        return {"sensors": [], "uav_specifications": []}

def main():
    # Centered title
    st.markdown("<h1 style='text-align: center;'>UAV Sensor Detection Probability Calculator</h1>", unsafe_allow_html=True)
    
    # Initialize session state for storing data across interactions
    if 'area_selected' not in st.session_state:
        st.session_state.area_selected = False
    if 'sensors' not in st.session_state:
        st.session_state.sensors = []
    if 'protected_areas' not in st.session_state:
        st.session_state.protected_areas = []
    if 'grid_size' not in st.session_state:
        st.session_state.grid_size = 0.01  # Default grid size in degrees
    if 'boundary_type' not in st.session_state:
        st.session_state.boundary_type = "rectangle"  # Default boundary type
    if 'boundary_points' not in st.session_state:
        st.session_state.boundary_points = []  # For polygon boundaries
    if 'last_drawn_feature' not in st.session_state:
        st.session_state.last_drawn_feature = None  # Track last drawn feature
    if 'placement_mode' not in st.session_state:
        st.session_state.placement_mode = False
    if 'current_sensor_location' not in st.session_state:
        st.session_state.current_sensor_location = None
    if 'potential_locations' not in st.session_state:
        st.session_state.potential_locations = []  # Initialize potential_locations
    
    # Initialize session state for sensor specifications
    if 'sensor_specifications' not in st.session_state:
        st.session_state.sensor_specifications = []
    
    # Initialize the location search session state
    if 'location_selected' not in st.session_state:
        st.session_state.location_selected = False
    
    # Default map center (not setting a default location)
    if 'map_center' not in st.session_state:
        st.session_state.map_center = None
    
    # Location Search Section - Only show if location not yet selected
    if not st.session_state.location_selected:
        # Create a centered container for the search UI
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Center-aligned header
            st.markdown("<h1 style='text-align: center;'>Search for a Location</h1>", unsafe_allow_html=True)

            # User input for location search with centered label
            st.markdown("<p style='text-align: center;'>Enter a location (city, address, landmark, etc.)</p>", unsafe_allow_html=True)

            # Create columns to control the width of the input box
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                location_input = st.text_input("Location", "", label_visibility="collapsed")

            # Search button centered
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_b:
                search_button = st.button("Search Location")
        
        # Handle location search button click
        if search_button and location_input:
            coordinates = geocode_location(location_input)
            
            with col2:
                if coordinates:
                    st.session_state.map_center = coordinates
                    st.session_state.location_selected = True
                    st.success(f"Location found: {location_input} at coordinates {coordinates}")
                    st.rerun()  # Reload the app to show the map
                else:
                    st.error(f"Could not find coordinates for '{location_input}'. Please try a different location.")
        
        # Skip the rest of the app until a location is selected
        return
    
    # Once a location is selected, show a button to change location
    st.sidebar.subheader("Current Location")
    st.sidebar.info(f"Centered at: {st.session_state.map_center[0]:.4f}, {st.session_state.map_center[1]:.4f}")
    
    if st.sidebar.button("Change Location"):
        st.session_state.location_selected = False
        st.rerun()
    
    # Sidebar for input parameters
    with st.sidebar:
        st.header("Configuration Parameters")
        
        # Add Sensor Information section - MOVED TO TOP
        st.subheader("Sensor Selection")
        
        # Load sensor data from JSON file
        sensor_data = load_sensor_data()
        
        # Create tabs for different sensor types
        if sensor_data["sensors"]:
            sensor_types = [sensor["sensor_type"] for sensor in sensor_data["sensors"]]
            sensor_tabs = st.tabs(sensor_types)
            
            # Display each sensor type in its own tab
            for i, tab in enumerate(sensor_tabs):
                with tab:
                    st.write(f"### {sensor_types[i]} Sensors")
                    
                    # Get parameters for this sensor type
                    sensors_of_type = sensor_data["sensors"][i]["parameters"]
                    
                    # Create an expander for each sensor model
                    for sensor in sensors_of_type:
                        with st.expander(f"{sensor['model']} - {sensor['manufacturer']}"):
                            # Display sensor details
                            st.write(f"**Description:** {sensor['description']}")
                            st.write(f"**Detection Range:** {sensor['detection_range']} meters")
                            st.write(f"**Response Time:** {sensor['response_time']} seconds")
                            st.write(f"**Price:** ${sensor['price_per_unit']:,}")
                            
                            # Display additional specifications in a more compact format
                            if "sensor_specifications" in sensor:
                                specs = sensor["sensor_specifications"]
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Specifications:**")
                                    for key, value in list(specs.items())[:len(specs)//2]:
                                        formatted_key = key.replace("_", " ").title()
                                        st.write(f"- {formatted_key}: {value}")
                                
                                with col2:
                                    st.write("&nbsp;")  # Empty header for alignment
                                    for key, value in list(specs.items())[len(specs)//2:]:
                                        formatted_key = key.replace("_", " ").title()
                                        st.write(f"- {formatted_key}: {value}")
                            
                            # Button to add this sensor to selection
                            if st.button("Add Sensor", key=f"add_{sensor['model']}"):
                                new_sensor = {
                                    "type": sensor_types[i],
                                    "detection_range": sensor["detection_range"],
                                    "response_time": sensor["response_time"],
                                    "model": sensor["model"],
                                    "manufacturer": sensor["manufacturer"],
                                    "price_per_unit": sensor["price_per_unit"],
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                                
                                # Add to session state
                                st.session_state.sensor_specifications.append(new_sensor)
                                st.success(f"Added {sensor['model']} sensor!")
                                st.rerun()  # Rerun to update the UI
        else:
            st.error("No sensor data found. Please check sensor-data.json file.")
        
        # Display the list of added sensors
        if st.session_state.sensor_specifications:
            st.subheader("Selected Sensors")
            
            # Display each sensor with a remove button
            for i, sensor in enumerate(st.session_state.sensor_specifications):
                with st.container():
                    st.markdown(f"**{i+1}. {sensor['type']}** - {sensor['model']}")
                    st.markdown(f"Range: {sensor['detection_range']}m, Response: {sensor['response_time']}s")
                    st.markdown(f"Price: ${sensor['price_per_unit']:.2f}")  # Display the price
                    
                    # Button to remove this sensor
                    if st.button("Remove", key=f"remove_sensor_{i}"):
                        st.session_state.sensor_specifications.pop(i)
                        st.success("Sensor removed!")
                        st.rerun()
                
                st.markdown("---")
        
        # Add UAV specifications - MOVED TO MIDDLE
        st.subheader("UAV Specifications")
        
        # If UAV specifications exist in the JSON file, display as a selectbox
        if "uav_specifications" in sensor_data and sensor_data["uav_specifications"]:
            uav_types = [uav["uav_type"] for uav in sensor_data["uav_specifications"]]
            selected_uav_type = st.selectbox("UAV Type", uav_types)
            
            # Find the selected UAV in the data
            selected_uav = next((uav for uav in sensor_data["uav_specifications"] if uav["uav_type"] == selected_uav_type), None)
            
            if selected_uav:
                # Display altitude range for the selected UAV
                altitude_range = selected_uav["altitude_range"]
                uav_altitude = st.slider("UAV Altitude (meters)", 
                                         min_value=float(altitude_range[0]), 
                                         max_value=float(altitude_range[1]),
                                         value=float((altitude_range[0] + altitude_range[1]) / 2))
                
                # Display speed range for the selected UAV
                speed_range = selected_uav["speed_range"]
                uav_speed = st.slider("UAV Speed (m/s)", 
                                     min_value=float(speed_range[0]),
                                     max_value=float(speed_range[1]),
                                     value=float((speed_range[0] + speed_range[1]) / 2))
                
                # Display additional UAV info
                st.info(f"Endurance: {selected_uav['endurance']} minutes\nPayload Capacity: {selected_uav['payload_capacity']} kg")
        else:
            # Fallback to manual input if no UAV specs in JSON
            uav_altitude = st.number_input("UAV Altitude (meters)", value=100.0, step=10.0)
            uav_speed = st.number_input("UAV Speed (m/s)", value=10.0, step=1.0)
        
        # Option to download all input information as GeoJSON
        if st.session_state.sensor_specifications:
            st.subheader("Export All Input Data")
            
            # Create export data
            export_data = export_all_data_to_geojson(
                boundary_type=st.session_state.boundary_type,
                boundary_points=st.session_state.boundary_points,
                sw_corner=st.session_state.get('sw_corner'),
                ne_corner=st.session_state.get('ne_corner'),
                sensor_locations=st.session_state.potential_locations,
                protected_areas=st.session_state.protected_areas,
                sensor_specifications=st.session_state.sensor_specifications,
                uav_specs={
                    "altitude": uav_altitude,
                    "speed": uav_speed
                }
            )
            
            # Convert to JSON string
            export_json = json.dumps(export_data, indent=2)
            
            # Create a download button for the JSON
            st.download_button(
                label="Download as GeoJSON",
                data=export_json,
                file_name="uav_sensor_project_data.geojson",
                mime="application/geo+json"
            )
    
    # Create tabs for different sections of the application
    # CHANGE: Reordered the tabs to match the new workflow
    # Create tabs for different sections of the application
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Map & Selection", "Coverage Areas", "Possible Sensor Placement", "Preprocess Detection Probability", "Prediction", "Optimization"])
    
    # Now use the dedicated functions for each tab
    with tab1:
        map_selection_tab()
    
    with tab2:
        protected_areas_tab()
    
    with tab3:
        sensor_tab()
    
    with tab4:
        detection_probability_tab()

    with tab5:
        st.header("Prediction")
        st.info("This feature is coming in the next implementation step.")
        
        # Placeholder for prediction calculations
        if st.session_state.area_selected and st.session_state.potential_locations:
            st.success("Area and sensors have been defined. Prediction functionality will be implemented here.")
        else:
            st.warning("Please define an area of interest and place sensors before making predictions.")

    with tab6:
        st.header("Optimization")
        st.info("This feature is coming in the next implementation step.")
        
        # Placeholder for optimization calculations
        if st.session_state.area_selected and st.session_state.potential_locations:
            st.success("Area and sensors have been defined. Optimization algorithm will be implemented here.")
            
            # Add a simple framework for what will be implemented
            st.subheader("Planned Optimization Features:")
            st.markdown("""
            - Sensor placement optimization based on coverage area
            - Cost optimization for sensor deployment
            - Detection probability optimization
            - Multi-objective optimization considering all constraints
            """)
        else:
            st.warning("Please define an area of interest and place sensors before running optimization.")

def export_all_data_to_geojson(boundary_type, boundary_points, sw_corner, ne_corner, 
                              sensor_locations, protected_areas, sensor_specifications,
                              uav_specs):
    """
    Export all user input data to a comprehensive GeoJSON structure
    """
    # Initialize the GeoJSON structure with reordered metadata
    geojson_data = {
        "type": "Input Data Collection",
        "features": [],
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sensor_specifications": sensor_specifications,  # First
            "uav_specifications": uav_specs                 # Second
        }
    }
    
    # Add the boundary area
    if boundary_type == "rectangle" and sw_corner and ne_corner:
        # Create a rectangle feature
        rectangle_coords = [
            [sw_corner[1], sw_corner[0]],  # SW corner as [lng, lat]
            [ne_corner[1], sw_corner[0]],  # SE corner
            [ne_corner[1], ne_corner[0]],  # NE corner
            [sw_corner[1], ne_corner[0]],  # NW corner
            [sw_corner[1], sw_corner[0]]   # Back to SW to close the polygon
        ]
        
        boundary_feature = {
            "type": "Feature",
            "properties": {
                "featureType": "boundary",
                "boundaryType": "rectangle"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [rectangle_coords]
            }
        }
        geojson_data["features"].append(boundary_feature)
    
    elif boundary_type == "polygon" and boundary_points:
        # Convert polygon points from [lat, lng] to [lng, lat] for GeoJSON
        polygon_coords = [[point[1], point[0]] for point in boundary_points]
        
        # Ensure the polygon is closed
        if polygon_coords[0] != polygon_coords[-1]:
            polygon_coords.append(polygon_coords[0])
        
        boundary_feature = {
            "type": "Feature",
            "properties": {
                "featureType": "boundary",
                "boundaryType": "polygon"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon_coords]
            }
        }
        geojson_data["features"].append(boundary_feature)
    
    # Add sensor locations
    for i, location in enumerate(sensor_locations):
        # Find matching sensor specification if available
        sensor_spec = None
        if i < len(sensor_specifications):
            sensor_spec = sensor_specifications[i]
        
        sensor_feature = {
            "type": "Feature",
            "properties": {
                "featureType": "sensor",
                "id": i + 1,
                "timestamp": location.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "specifications": sensor_spec
            },
            "geometry": {
                "type": "Point",
                "coordinates": [location['lng'], location['lat']]
            }
        }
        geojson_data["features"].append(sensor_feature)
    
    # Add protected areas
    for i, area in enumerate(protected_areas):
        # Convert points from [lat, lng] to [lng, lat] for GeoJSON
        area_coords = [[point[1], point[0]] for point in area['points']]
        
        # Ensure the polygon is closed
        if area_coords[0] != area_coords[-1]:
            area_coords.append(area_coords[0])
        
        area_feature = {
            "type": "Feature",
            "properties": {
                "featureType": "protectedArea",
                "id": i + 1,
                "name": area.get('name', f"Protected Area {i+1}"),
                "timestamp": area.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [area_coords]
            }
        }
        geojson_data["features"].append(area_feature)
    
    return geojson_data

if __name__ == "__main__":
    main()