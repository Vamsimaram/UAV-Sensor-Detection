import streamlit as st
import folium
from streamlit_folium import folium_static, st_folium
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
from pymongo import MongoClient
from folium.plugins import Draw

# Set up the Streamlit page configuration
st.set_page_config(page_title="UAV Sensor Detection Probability Calculator", 
                   layout="wide", 
                   initial_sidebar_state="expanded")

def create_grid_overlay(map_obj, sw_corner, ne_corner, grid_size):
    """
    Create a grid overlay on the map based on specified corners and grid size
    
    Parameters:
    -----------
    map_obj : folium.Map
        The map object to add the grid to
    sw_corner : list
        [lat, lng] of southwest corner
    ne_corner : list
        [lat, lng] of northeast corner
    grid_size : float
        Size of grid squares in degrees
    """
    # Calculate number of rows and columns in the grid
    lat_range = ne_corner[0] - sw_corner[0]
    lng_range = ne_corner[1] - sw_corner[1]
    
    num_rows = int(lat_range / grid_size) + 1
    num_cols = int(lng_range / grid_size) + 1
    
    # Create grid cell for each row and column
    for i in range(num_rows):
        for j in range(num_cols):
            lat_sw = sw_corner[0] + i * grid_size
            lng_sw = sw_corner[1] + j * grid_size
            
            lat_ne = lat_sw + grid_size
            lng_ne = lng_sw + grid_size
            
            # Add rectangle for each grid cell with more visible styling
            folium.Rectangle(
                bounds=[[lat_sw, lng_sw], [lat_ne, lng_ne]],
                color='black',  # Changed from blue to black
                weight=2,       # Increased from 1 to 2
                fill=False,
                opacity=0.8,    # Increased from 0.5 to 0.8
                popup=f"Grid Cell ({i},{j})"
            ).add_to(map_obj)

def create_boundary_drawing_map(center, zoom_start=12, predefined_locations=None):
    """
    Create a map with drawing controls for boundary selection
    
    Parameters:
    -----------
    center : list
        [lat, lng] center of the map
    zoom_start : int
        Initial zoom level
    predefined_locations : dict
        Dictionary of predefined locations to add to the map
        
    Returns:
    --------
    folium.Map : Map object with drawing controls
    """
    m = folium.Map(location=center, zoom_start=zoom_start)
    
    # Add the draw control to the map with enhanced options
    draw = Draw(
        export=False,  # Disable the built-in export button
        position='topright',  # Position the control on the top right
        draw_options={
            'polyline': False,
            'circle': False,
            'marker': False,
            'circlemarker': False,
            'polygon': {
                'allowIntersection': False,  # Prevents self-intersections
                'drawError': {
                    'color': '#e1e100',
                    'message': 'Self-intersection not allowed!'
                },
                'shapeOptions': {
                    'color': '#ff0000',  # Red outline
                    'fillColor': '#ff6666',  # Lighter red fill
                    'fillOpacity': 0.5
                }
            },
            'rectangle': {
                'shapeOptions': {
                    'color': '#ff0000',  # Red outline
                    'fillColor': '#ff6666',  # Lighter red fill
                    'fillOpacity': 0.5
                }
            }
        },
        edit_options={
            'featureGroup': None,
            'poly': {
                'allowIntersection': False
            }
        }
    )
    draw.add_to(m)
    
    # Add instructions as a map control
    instructions_html = """
    <div style="position: fixed; 
                bottom: 50px; 
                left: 50px; 
                width: 300px;
                height: auto;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 0 5px rgba(0,0,0,0.5);
                padding: 10px;
                z-index: 1000;">
        <h4 style="margin-top: 0;">Drawing Instructions:</h4>
        <ul style="padding-left: 20px; margin-bottom: 0;">
            <li>Click the rectangle or polygon tool on the top right</li>
            <li>Draw your area on the map</li>
            <li>The coordinates will be captured automatically</li>
            <li>Click "Set Drawn Area" when finished</li>
            <li>Use the Download button below the map to save your boundary</li>
        </ul>
    </div>
    """
    
    # Add the instructions to the map
    instructions = folium.Element(instructions_html)
    m.get_root().html.add_child(instructions)
    
    return m

def is_point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using ray casting algorithm
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def is_point_in_rectangle(point, sw_corner, ne_corner):
    """
    Check if a point is inside a rectangle
    """
    lat, lng = point
    return (sw_corner[0] <= lat <= ne_corner[0]) and (sw_corner[1] <= lng <= ne_corner[1])

def map_selection_tab():
    """
    Map-based selection tab for defining area of interest
    """
    st.header("Select Area of Interest")
    
    # Only provide "Draw on Map" option
    st.subheader("Draw Boundary on Map")
    
    # Show appropriate guidance based on whether an area is already selected
    if st.session_state.area_selected:
        st.info("Your current selected area is shown in red. You can modify it by drawing a new boundary and clicking 'Set Drawn Area'.")
    else:
        st.info("Use the drawing tools (rectangle or polygon) in the top right corner of the map to create your boundary. Then click 'Set Drawn Area'.")
    
    # Create a map with drawing controls
    drawing_map = create_boundary_drawing_map(
        st.session_state.map_center,
        zoom_start=12,
        predefined_locations={}  # Empty dictionary as we're not using predefined locations
    )
    
    # If an area is already selected, show it on the drawing map
    if st.session_state.area_selected:
        if st.session_state.boundary_type == "polygon":
            # Draw the selected area on the map
            folium.Polygon(
                locations=st.session_state.boundary_points,
                color='red',
                fill=True,
                fill_opacity=0.2,
                tooltip="Current Selected Area"
            ).add_to(drawing_map)
        elif st.session_state.boundary_type == "rectangle":
            # Draw rectangle boundary
            folium.Rectangle(
                bounds=[[st.session_state.sw_corner[0], st.session_state.sw_corner[1]], 
                      [st.session_state.ne_corner[0], st.session_state.ne_corner[1]]],
                color='red',
                fill=True,
                fill_opacity=0.2,
                tooltip="Current Selected Area"
            ).add_to(drawing_map)
    
    # Display the map with drawing controls and capture the response
    map_data = st_folium(drawing_map, width=1000, height=500)
    
    # Process drawn features from map_data
    if map_data and 'all_drawings' in map_data and map_data['all_drawings']:
        st.session_state.last_drawn_feature = map_data['all_drawings']
        
        # Extract coordinates from the last drawing
        if len(map_data['all_drawings']) > 0:
            last_drawing = map_data['all_drawings'][-1]
            
            if 'geometry' in last_drawing and 'coordinates' in last_drawing['geometry']:
                coords = last_drawing['geometry']['coordinates']
                
                # Handle different geometry types (polygon vs rectangle)
                if last_drawing['geometry']['type'] == 'Polygon':
                    # For polygons, coordinates are in the format [[[lng1, lat1], [lng2, lat2], ...]]
                    # Extract the first ring of coordinates
                    if coords and coords[0]:
                        boundary_points = []
                        for coord in coords[0]:
                            # Convert [lng, lat] to [lat, lng] format
                            boundary_points.append([coord[1], coord[0]])
                        
                        # Update session state with the new boundary points
                        st.session_state.boundary_points = boundary_points
                        st.session_state.boundary_type = "polygon"
                        
                        # Calculate SW and NE corners for grid creation
                        lats = [p[0] for p in boundary_points]
                        lngs = [p[1] for p in boundary_points]
                        st.session_state.sw_corner = [min(lats), min(lngs)]
                        st.session_state.ne_corner = [max(lats), max(lngs)]
                        
                        # Update map center
                        st.session_state.map_center = [(min(lats) + max(lats))/2, 
                                                    (min(lngs) + max(lngs))/2]
                # Handle rectangle type specifically - this can also be identified as a Polygon
                elif last_drawing['geometry']['type'] == 'Rectangle' or (
                    last_drawing['geometry']['type'] == 'Polygon' and len(coords[0]) == 5):
                    # For rectangles, extract the boundary points
                    if coords and coords[0]:
                        boundary_points = []
                        for coord in coords[0]:
                            # Convert [lng, lat] to [lat, lng] format
                            boundary_points.append([coord[1], coord[0]])
                        
                        # Calculate SW and NE corners
                        lats = [p[0] for p in boundary_points]
                        lngs = [p[1] for p in boundary_points]
                        sw_corner = [min(lats), min(lngs)]
                        ne_corner = [max(lats), max(lngs)]
                        
                        # Update session state
                        st.session_state.sw_corner = sw_corner
                        st.session_state.ne_corner = ne_corner
                        st.session_state.boundary_type = "rectangle"
                        st.session_state.boundary_points = boundary_points
                        
                        # Update map center
                        st.session_state.map_center = [(min(lats) + max(lats))/2, 
                                                    (min(lngs) + max(lngs))/2]
    
    # Button to set the drawn area
    if st.button("Set Drawn Area"):
        if not st.session_state.last_drawn_feature:
            st.error("Please draw an area on the map first.")
        else:
            st.session_state.area_selected = True
            
            # Show the selected area on a new map
            selected_area_map = folium.Map(
                location=st.session_state.map_center, 
                zoom_start=12
            )
            
            if st.session_state.boundary_type == "polygon":
                # Draw the selected area on the map with a different color
                folium.Polygon(
                    locations=st.session_state.boundary_points,
                    color='red',
                    fill=True,
                    fill_opacity=0.4,
                    tooltip="Selected Area"
                ).add_to(selected_area_map)
            elif st.session_state.boundary_type == "rectangle":
                # Draw rectangle boundary
                folium.Rectangle(
                    bounds=[[st.session_state.sw_corner[0], st.session_state.sw_corner[1]], 
                        [st.session_state.ne_corner[0], st.session_state.ne_corner[1]]],
                    color='red',
                    fill=True,
                    fill_opacity=0.4,
                    tooltip="Selected Area"
                ).add_to(selected_area_map)
            
            # Display the map with the selected area
            st.subheader("Your Selected Area")
            folium_static(selected_area_map)
            
            st.success("Drawn area set successfully! The highlighted area will appear in all tabs.")
    
    if st.button("Clear Drawn Areas"):
        st.session_state.area_selected = False
        st.session_state.boundary_type = "rectangle"
        st.session_state.boundary_points = []
        st.session_state.last_drawn_feature = None
        st.success("Drawn areas cleared.")
    
    # Grid size input (we'll keep this since it's needed for some functionality)
    grid_size = st.slider("Grid Size (degrees)", min_value=0.001, max_value=0.05, value=0.01, step=0.001)
    st.session_state.grid_size = grid_size
    
    # Display the current boundary points if available and add download option
    if st.session_state.boundary_points:
        st.subheader("Current Boundary Points")
        points_df = pd.DataFrame(st.session_state.boundary_points, columns=["Latitude", "Longitude"])
        st.dataframe(points_df)
        
        # Add option to download as GeoJSON only
        import json
        
        @st.cache_data
        def convert_to_geojson(points):
            features = []
            # Create a feature for the polygon
            coordinates = [[point[1], point[0]] for point in points]  # Convert [lat, lng] to [lng, lat] for GeoJSON
            # Close the polygon by adding the first point at the end
            if coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])
            feature = {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates]
                }
            }
            features.append(feature)
            
            geojson = {
                "type": "FeatureCollection",
                "features": features
            }
            return json.dumps(geojson, indent=2)
        
        geojson_str = convert_to_geojson(st.session_state.boundary_points)
        st.download_button(
            label="Download Boundary as GeoJSON",
            data=geojson_str,
            file_name='boundary.geojson',
            mime='application/json',
        )

def sensor_tab():
    """
    Map-based point selection for potential sensor locations using the Draw plugin
    """
    st.header("Potential Sensor Locations")
    
    if not st.session_state.area_selected:
        st.warning("Please select an area of interest in the Map & Selection tab first.")
        return
    
    # Initialize session state for potential locations if not already done
    if 'potential_locations' not in st.session_state:
        st.session_state.potential_locations = []
    
    # Ensure we have json imported
    import json
    
    # Create columns for layout - using different ratio for better layout
    map_col, list_col = st.columns([4, 1])
    
    # Display list of marked locations in the right column
    with list_col:
        st.subheader("Marked Points")
        
        # Show total count
        if st.session_state.potential_locations:
            st.info(f"Total points: {len(st.session_state.potential_locations)}")
            
            # Create a scrollable container for the points list
            points_container = st.container()
            with points_container:
                # Use a more efficient approach for displaying points
                for i, location in enumerate(st.session_state.potential_locations):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**Point {i+1}:** [{location['lat']:.6f}, {location['lng']:.6f}]")
                    with col2:
                        # Add a remove button for each point
                        if st.button("🗑️", key=f"remove_point_{i}"):
                            st.session_state.potential_locations.pop(i)
                            # Clear map data to prevent issues with point indexing
                            st.session_state.map_data = None
                            st.rerun()
            
            # MOVED: Export and Clear buttons now appear below point coordinates
            if st.session_state.potential_locations:
                st.markdown("---")
                
                # Create the GeoJSON data
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": []
                }
                
                for location in st.session_state.potential_locations:
                    feature = {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Point",
                            "coordinates": [location['lng'], location['lat']]
                        }
                    }
                    geojson_data["features"].append(feature)
                
                # Convert to JSON string
                geojson_str = json.dumps(geojson_data, indent=2)
                
                # Export button
                st.download_button(
                    label="📋 Export GeoJSON",
                    data=geojson_str,
                    file_name='sensor_points.geojson',
                    mime='application/geo+json',
                    key="export_button_in_list"
                )
                
                # Clear button
                if st.button("🗑️ Clear All", key="clear_button_in_list"):
                    st.session_state.potential_locations = []
                    st.session_state.map_data = None
                    st.rerun()
        else:
            st.info("No points marked yet.")
            st.markdown("""
            **Instructions:**
            1. Click the marker tool in the top right corner of the map
            2. Place ONE marker at a time within the red highlighted area
            3. Click "Set Drawn Sensors" after placing EACH marker
            4. Export the points when finished
            """)
    
    with map_col:
        st.subheader("Mark Potential Sensor Locations")
        st.warning("IMPORTANT: Place ONE point at a time and when you select all points. click 'Set Drawn Sensors'. Points outside the highlighted area will be ignored.")
        
        # Create map centered on the selected area with drawing controls
        m = folium.Map(location=st.session_state.map_center, zoom_start=14)
        
        # Draw the selected area boundary with more visible highlighting
        if st.session_state.boundary_type == "rectangle":
            folium.Rectangle(
                bounds=[[st.session_state.sw_corner[0], st.session_state.sw_corner[1]], 
                       [st.session_state.ne_corner[0], st.session_state.ne_corner[1]]],
                color='red',
                weight=3,
                fill=True,
                fill_color='red',
                fill_opacity=0.2,
                popup="Selected Area"
            ).add_to(m)
        elif st.session_state.boundary_type == "polygon":
            folium.Polygon(
                locations=st.session_state.boundary_points,
                color='red',
                weight=3,
                fill=True,
                fill_color='red',
                fill_opacity=0.2,
                popup="Selected Area"
            ).add_to(m)
        
        # Display all marked potential locations with highly visible markers
        for i, location in enumerate(st.session_state.potential_locations):
            folium.Marker(
                location=[location['lat'], location['lng']],
                popup=f"Point {i+1}: [{location['lat']:.6f}, {location['lng']:.6f}]",
                tooltip=f"Point {i+1}",
                icon=folium.Icon(color='blue', icon='circle', prefix='fa')
            ).add_to(m)
        
        # Add the Draw control to the map - but only enable markers
        draw = Draw(
            export=False,  # Disable export to prevent duplications
            position='topright',
            draw_options={
                'polyline': False,
                'polygon': False,
                'rectangle': False,
                'circle': False,
                'circlemarker': False,
                'marker': True,  # Simplified marker configuration
            },
            edit_options={
                'featureGroup': None,
                'remove': True  # Allow removal of markers directly from the map
            }
        )
        draw.add_to(m)
        
        # Add instructions as a map control
        instructions_html = """
        <div style="position: fixed; 
                    bottom: 50px; 
                    left: 50px; 
                    width: 300px;
                    height: auto;
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 0 5px rgba(0,0,0,0.5);
                    padding: 10px;
                    z-index: 1000;">
            <h4 style="margin-top: 0;">Sensor Placement Instructions:</h4>
            <ul style="padding-left: 20px; margin-bottom: 0;">
                <li>Click the marker tool in the top right corner</li>
                <li>Place ONE point at a time on the map</li>
                <li>Click "Set Drawn Sensors" after EACH point</li>
                <li>Points must be inside the red highlighted area</li>
                <li>Use the Export button to save your sensor locations</li>
            </ul>
            <p style="color: red; font-weight: bold; margin-top: 5px; margin-bottom: 0;">
                Note: Points outside the highlighted area will be ignored!
            </p>
        </div>
        """
        instructions = folium.Element(instructions_html)
        m.get_root().html.add_child(instructions)
        
        # Display the map and get drawing data
        map_data = st_folium(m, width=800, height=600, key="sensor_map_display")
        
        # Store the map data in session state to prevent it from disappearing on rerun
        if map_data:
            st.session_state.map_data = map_data
        
        # Use the stored map data if it exists
        if 'map_data' not in st.session_state:
            st.session_state.map_data = None
        
        # Button to set drawn sensors
        if st.button("Set Drawn Sensors", key="set_sensors_button"):
            if st.session_state.map_data and 'all_drawings' in st.session_state.map_data and st.session_state.map_data['all_drawings']:
                new_sensors = []
                invalid_points = 0
                existing_locations = {(loc['lat'], loc['lng']) for loc in st.session_state.potential_locations}
                
                for feature in st.session_state.map_data['all_drawings']:
                    if feature['geometry']['type'] == 'Point':
                        lng, lat = feature['geometry']['coordinates']
                        
                        # Check if the point is inside the selected area
                        is_inside = False
                        if st.session_state.boundary_type == "rectangle":
                            is_inside = is_point_in_rectangle(
                                [lat, lng], 
                                st.session_state.sw_corner, 
                                st.session_state.ne_corner
                            )
                        elif st.session_state.boundary_type == "polygon":
                            is_inside = is_point_in_polygon(
                                [lat, lng], 
                                st.session_state.boundary_points
                            )
                        
                        if is_inside and (lat, lng) not in existing_locations:
                            new_sensor = {
                                'lat': lat,
                                'lng': lng,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            new_sensors.append(new_sensor)
                            existing_locations.add((lat, lng))
                        else:
                            if not is_inside:
                                invalid_points += 1
                
                # Add new sensors to the potential locations
                if new_sensors:
                    st.session_state.potential_locations.extend(new_sensors)
                    
                    if invalid_points > 0:
                        st.warning(f"Added {len(new_sensors)} new sensor locations. {invalid_points} point(s) were outside the selected region and were ignored.")
                    else:
                        st.success(f"Added {len(new_sensors)} new sensor locations!")
                    
                    # Important: Clear the map_data after adding points to allow for new points to be added
                    st.session_state.map_data = None
                    
                    st.rerun()
                else:
                    if invalid_points > 0:
                        st.warning(f"No points were added. All {invalid_points} point(s) were outside the selected region and were ignored.")
                    else:
                        st.info("No new valid sensor locations were marked.")
            else:
                st.warning("No markers detected. Please place markers on the map first.")

def protected_areas_tab():
    # Protected areas functionality
    st.header("Protected Areas")
    
    if not st.session_state.area_selected:
        st.warning("Please select an area of interest in the first tab before defining protected areas.")
        return
    
    # Display the map with the selected area
    m = folium.Map(location=st.session_state.map_center, zoom_start=12)
    
    # Draw the selected area boundary
    if st.session_state.boundary_type == "rectangle":
        # Draw rectangle boundary
        folium.Rectangle(
            bounds=[[st.session_state.sw_corner[0], st.session_state.sw_corner[1]], 
                   [st.session_state.ne_corner[0], st.session_state.ne_corner[1]]],
            color='red',
            fill=True,
            fill_opacity=0.2,
            popup="Selected Area",
            tooltip="Selected Area"
        ).add_to(m)
    elif st.session_state.boundary_type == "polygon":
        # Draw polygon boundary
        folium.Polygon(
            locations=st.session_state.boundary_points,
            color='red',
            fill=True,
            fill_opacity=0.2,
            popup="Selected Area",
            tooltip="Selected Area"
        ).add_to(m)
    
    # Draw the grid if using rectangle boundary
    if st.session_state.boundary_type == "rectangle":
        create_grid_overlay(m, st.session_state.sw_corner, st.session_state.ne_corner, st.session_state.grid_size)
    
    # Display existing protected areas
    for i, area in enumerate(st.session_state.protected_areas):
        folium.Polygon(
            locations=area['points'],
            popup=f"Protected Area {i+1}: {area['name']}",
            tooltip=f"Protected Area {i+1}: {area['name']}",
            color='purple',
            fill=True,
            fill_opacity=0.4
        ).add_to(m)
    
    # Display the map
    folium_static(m)
    
    # Form to add new protected areas
    with st.form("add_protected_area_form"):
        st.subheader("Add New Protected Area")
        
        area_name = st.text_input("Area Name")
        
        st.markdown("Enter coordinates for the polygon vertices (minimum 3 points)")
        
        num_points = st.number_input("Number of Points", min_value=3, max_value=10, value=3)
        
        points = []
        for i in range(num_points):
            st.markdown(f"**Point {i+1}**")
            col1, col2 = st.columns(2)
            with col1:
                point_lat = st.number_input(f"Latitude {i+1}", value=st.session_state.map_center[0], step=0.001, key=f"lat_{i}")
            with col2:
                point_lng = st.number_input(f"Longitude {i+1}", value=st.session_state.map_center[1], step=0.001, key=f"lng_{i}")
            points.append([point_lat, point_lng])
        
        # Submit button
        submitted = st.form_submit_button("Add Protected Area")
        
        if submitted:
            if len(points) < 3:
                st.error("A protected area must have at least 3 points.")
            else:
                # Add the new protected area to the list
                new_area = {
                    'name': area_name,
                    'points': points
                }
                st.session_state.protected_areas.append(new_area)
                st.success(f"Protected area added successfully! Total areas: {len(st.session_state.protected_areas)}")
    
    # Show current protected areas
    if st.session_state.protected_areas:
        st.subheader("Current Protected Areas")
        
        for i, area in enumerate(st.session_state.protected_areas):
            st.markdown(f"**Area {i+1}**: {area['name']} - {len(area['points'])} points")
        
        # Option to remove protected areas
        if st.button("Remove All Protected Areas"):
            st.session_state.protected_areas = []
            st.success("All protected areas removed!")

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