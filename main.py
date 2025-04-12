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
    
    # Sidebar for input parameters
    with st.sidebar:
        st.header("Input Parameters")
        
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
        
        # Default map center (Washington DC area as an example)
        if 'map_center' not in st.session_state:
            st.session_state.map_center = [38.9072, -77.0369]
        
        # Add UAV specifications
        st.subheader("UAV Specifications")
        uav_altitude = st.number_input("UAV Altitude (meters)", value=100.0, step=10.0)
        uav_speed = st.number_input("UAV Speed (m/s)", value=10.0, step=1.0)
        
        # Add environmental conditions
        st.subheader("Environmental Conditions")
        weather_condition = st.selectbox("Weather", ["Clear", "Fog", "Rain", "Snow"])
    
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
        if st.session_state.area_selected and st.session_state.sensors:
            st.success("Area and sensors have been defined. Detection probability calculations will be implemented here.")
        else:
            st.warning("Please define an area of interest and place sensors before calculating detection probabilities.")

if __name__ == "__main__":
    main()