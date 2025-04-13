import streamlit as st
import folium
from streamlit_folium import folium_static, st_folium
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
from pymongo import MongoClient

# Import functions from other modules
from map_utils import create_grid_overlay, create_boundary_drawing_map, is_point_in_polygon, is_point_in_rectangle
from tabs.map_selection import map_selection_tab
from tabs.sensor import sensor_tab
from tabs.protected_areas import protected_areas_tab

# Set up the Streamlit page configuration
st.set_page_config(page_title="UAV Sensor Detection Probability Calculator", 
                   layout="wide", 
                   initial_sidebar_state="expanded")

def main():
    st.title("UAV Sensor Detection Probability Calculator")
    
    # Initialize predefined locations for quick navigation
    predefined_locations = {
        "Washington DC": {"center": [38.9072, -77.0369]},
        "New York": {"center": [40.7128, -74.0060]},
        "San Francisco": {"center": [37.7749, -122.4194]},
        "Chicago": {"center": [41.8781, -87.6298]}
    }
    
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
    
    # Default map center (Washington DC area as an example)
    if 'map_center' not in st.session_state:
        st.session_state.map_center = [38.9072, -77.0369]
    
    # Sidebar for input parameters
    with st.sidebar:
        st.header("Input Parameters")
        
        # Add Sensor Information section - MOVED TO TOP
        st.subheader("Sensor Information")
        
        # Create an expander for adding new sensor specifications
        with st.expander("Add New Sensor Type", expanded=len(st.session_state.sensor_specifications) == 0):
            # Form for adding a new sensor
            sensor_type = st.selectbox("Sensor Type", ["Radar", "RF", "Lidar", "Infrared", "Acoustic", "Other"])
            
            # If "Other" is selected, allow custom input
            custom_sensor_type = None
            if sensor_type == "Other":
                custom_sensor_type = st.text_input("Specify Sensor Type")
            
            # Sensor characteristics
            detection_range = st.number_input("Detection Range (meters)", min_value=0.0, value=100.0, step=10.0)
            response_time = st.number_input("Response Time (seconds)", min_value=0.0, value=1.0, step=0.1)
            sensor_model = st.text_input("Sensor Model/Name (required)")
            
            # Button to add the sensor to the list
            if st.button("Add Sensor"):
                # Validate that the sensor model is provided
                if not sensor_model.strip():
                    st.error("Sensor Model/Name is required. Please enter a value.")
                # Check for duplicate sensor names
                elif any(sensor["model"].lower() == sensor_model.strip().lower() for sensor in st.session_state.sensor_specifications):
                    st.error(f"A sensor with the name '{sensor_model}' already exists. Please use a unique name.")
                else:
                    # Create the sensor specification dict
                    new_sensor = {
                        "type": custom_sensor_type if sensor_type == "Other" else sensor_type,
                        "detection_range": detection_range,
                        "response_time": response_time,
                        "model": sensor_model.strip(),  # Trim whitespace
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Add to session state
                    st.session_state.sensor_specifications.append(new_sensor)
                    st.success(f"Added {new_sensor['type']} sensor!")
                    st.rerun()  # Rerun to update the UI
        
        # Display the list of added sensors
        if st.session_state.sensor_specifications:
            st.subheader("Added Sensors")
            
            # Display each sensor with a remove button
            for i, sensor in enumerate(st.session_state.sensor_specifications):
                with st.container():
                    st.markdown(f"**{i+1}. {sensor['type']}** - {sensor['model']}")
                    st.markdown(f"Range: {sensor['detection_range']}m, Response: {sensor['response_time']}s")
                    
                    # Button to remove this sensor
                    if st.button("Remove", key=f"remove_sensor_{i}"):
                        st.session_state.sensor_specifications.pop(i)
                        st.success("Sensor removed!")
                        st.rerun()
                
                st.markdown("---")
        
        # Add UAV specifications - MOVED TO MIDDLE
        st.subheader("UAV Specifications")
        uav_altitude = st.number_input("UAV Altitude (meters)", value=100.0, step=10.0)
        uav_speed = st.number_input("UAV Speed (m/s)", value=10.0, step=1.0)
        
        # Add environmental conditions - MOVED TO BOTTOM
        st.subheader("Environmental Conditions")
        weather_condition = st.selectbox("Weather", ["Clear", "Fog", "Rain", "Snow"])
        
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
                },
                environmental_conditions={
                    "weather": weather_condition
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
    tab1, tab2, tab3, tab4 = st.tabs(["Map & Selection", "Sensor Placement", "Protected Areas", "Detection Probability"])
    
    # Now use the dedicated functions for each tab
    with tab1:
        map_selection_tab()
    
    with tab2:
        sensor_tab()
    
    with tab3:
        protected_areas_tab()
    
    with tab4:
        st.header("Detection Probability")
        st.info("This feature is coming in the next implementation step.")
        
        # Placeholder for detection probability calculations
        if st.session_state.area_selected and st.session_state.potential_locations:
            st.success("Area and sensors have been defined. Detection probability calculations will be implemented here.")
        else:
            st.warning("Please define an area of interest and place sensors before calculating detection probabilities.")

def export_all_data_to_geojson(boundary_type, boundary_points, sw_corner, ne_corner, 
                              sensor_locations, protected_areas, sensor_specifications,
                              uav_specs, environmental_conditions):
    """
    Export all user input data to a comprehensive GeoJSON structure
    """
    # Initialize the GeoJSON structure with reordered metadata
    geojson_data = {
        "type": "FeatureCollection",
        "features": [],
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sensor_specifications": sensor_specifications,  # First
            "uav_specifications": uav_specs,                # Second
            "environmental_conditions": environmental_conditions  # Third
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