import streamlit as st
import folium
from streamlit_folium import folium_static, st_folium
import pandas as pd
import json
import numpy as np
from shapely.geometry import Point, Polygon

# Import from map_utils
from map_utils import create_boundary_drawing_map

def draw_polygon_on_map(map_object, coordinates, color='red', fill_opacity=0.3, tooltip="Selected Area"):
    """
    Draw a polygon on a map with correctly ordered coordinates
    """
    # For polygons, connect points in the order they were entered
    folium_coords = [[point[0], point[1]] for point in coordinates]
    
    folium.Polygon(
        locations=folium_coords,
        color=color,
        fill=True,
        fill_opacity=fill_opacity,
        tooltip=tooltip
    ).add_to(map_object)

def create_true_square_grid(map_obj, sw_corner, ne_corner, grid_size_km):
    """
    Create a grid overlay on the map with true square cells (equal distances in all directions)
    
    Parameters:
    -----------
    map_obj : folium.Map
        The map object to add the grid to
    sw_corner : list
        [lat, lng] of southwest corner
    ne_corner : list
        [lat, lng] of northeast corner
    grid_size_km : float
        Size of grid squares in kilometers (same size in both dimensions)
    """
    # Earth's radius in kilometers
    earth_radius = 6371.0  # km
    
    # Convert grid size from km to degrees for latitude (roughly constant)
    # 1 degree of latitude is approximately 111 km
    lat_deg_per_km = 1 / 111.0
    grid_size_lat = grid_size_km * lat_deg_per_km
    
    # Get center latitude for longitude conversion (longitude degrees vary with latitude)
    center_lat = (sw_corner[0] + ne_corner[0]) / 2
    
    # Convert grid size from km to degrees for longitude at this latitude
    # 1 degree of longitude is approximately 111 * cos(latitude) km
    lng_deg_per_km = 1 / (111.0 * np.cos(np.radians(center_lat)))
    grid_size_lng = grid_size_km * lng_deg_per_km
    
    # Calculate the width and height of the area in kilometers
    width_km = (ne_corner[1] - sw_corner[1]) / lng_deg_per_km
    height_km = (ne_corner[0] - sw_corner[0]) / lat_deg_per_km
    
    # Calculate number of rows and columns needed to cover the area
    num_rows = int(np.ceil(height_km / grid_size_km))
    num_cols = int(np.ceil(width_km / grid_size_km))
    
    # Create grid cell for each row and column
    for i in range(num_rows):
        for j in range(num_cols):
            # Calculate the corners of this cell in degrees
            lat_sw = sw_corner[0] + i * grid_size_lat
            lng_sw = sw_corner[1] + j * grid_size_lng
            
            lat_ne = lat_sw + grid_size_lat
            lng_ne = lng_sw + grid_size_lng
            
            # Add rectangle for each grid cell
            folium.Rectangle(
                bounds=[[lat_sw, lng_sw], [lat_ne, lng_ne]],
                color='black',
                weight=1,
                fill=False,
                opacity=0.7,
                popup=f"Grid Cell ({i},{j})<br>ID: r{i}c{j}<br>Size: {grid_size_km:.2f}km × {grid_size_km:.2f}km"
            ).add_to(map_obj)

def map_selection_tab():
    """
    Map-based selection tab for defining area of interest
    """
    # Default starting coordinate (San Francisco)
    default_coordinate = [37.7749, -122.4194]
    
    st.header("Select Area of Interest")
    
    # Grid configuration section - Only show if an area is selected
    if st.session_state.area_selected:
        st.markdown("---")
        st.subheader("Grid Configuration")
        
        # Add grid configuration controls
        enable_grid = st.checkbox("Enable Grid Overlay", value=st.session_state.get('grid_enabled', False))
        
        if enable_grid:
            # Grid size control in degrees but display calculated km
            grid_size_degrees = st.number_input(
                "Grid Square Side Length (degrees)",
                min_value=0.0001,
                max_value=0.1, 
                value=st.session_state.get('grid_size_degrees', 0.01),
                format="%.4f",
                help="Size of each grid square in degrees. Will be converted to equal distances in all directions."
            )
            
            # Convert degrees to kilometers (approximate)
            # Earth's radius in kilometers
            earth_radius = 6371.0  # km
            
            # 1 degree of latitude is approximately 111 km
            grid_size_km = grid_size_degrees * 111.0
            
            # Store grid settings in session state
            st.session_state.grid_enabled = enable_grid
            st.session_state.grid_size_degrees = grid_size_degrees
            st.session_state.grid_size_km = grid_size_km
            
            # Calculate grid statistics
            if st.session_state.area_selected:
                sw_corner = st.session_state.sw_corner
                ne_corner = st.session_state.ne_corner
                
                # Get center latitude for calculations
                center_lat = (sw_corner[0] + ne_corner[0]) / 2
                
                # Calculate the width and height of the area in kilometers
                # Width: Convert longitude difference to km at this latitude
                width_km = (ne_corner[1] - sw_corner[1]) * 111.0 * np.cos(np.radians(center_lat))
                # Height: Convert latitude difference to km
                height_km = (ne_corner[0] - sw_corner[0]) * 111.0
                
                # Calculate number of cells in each dimension
                num_rows = int(np.ceil(height_km / grid_size_km))
                num_cols = int(np.ceil(width_km / grid_size_km))
                
                # Display statistics in an expander
                with st.expander("Grid Statistics", expanded=True):
                    st.write(f"Grid size: {grid_size_km:.2f} km × {grid_size_km:.2f} km (true squares)")
                    st.write(f"Input value: {grid_size_degrees:.4f} degrees")
                    st.write(f"Selected area: {width_km:.2f} km × {height_km:.2f} km")
                    st.write(f"Number of rows: {num_rows}")
                    st.write(f"Number of columns: {num_cols}")
                    st.write(f"Total grid cells: {num_rows * num_cols}")
        else:
            st.session_state.grid_enabled = False
    
    st.markdown("---")
    
    # Add a new tab selection for manual coordinate entry vs drawing
    selection_method = st.radio(
        "Select area by:",
        ["Drawing on map", "Entering coordinates"]
    )
    
    if selection_method == "Entering coordinates":
        st.subheader("Enter Coordinates")
        st.info("Enter coordinates in latitude, longitude format")
        
        # Initialize the coordinates list in session state if it doesn't exist
        if 'coord_input_list' not in st.session_state:
            st.session_state.coord_input_list = []
        
        # Display existing coordinates
        if st.session_state.coord_input_list:
            st.write("Current coordinate points:")
            coords_df = pd.DataFrame(
                st.session_state.coord_input_list, 
                columns=["Latitude", "Longitude"]
            )
            st.dataframe(coords_df, use_container_width=True)
        
        # Form for adding a new coordinate
        with st.form("add_coordinate_form"):
            st.write("Add a new coordinate point:")
            col1, col2 = st.columns(2)
            with col1:
                new_lat = st.number_input("Latitude", 
                                         value=default_coordinate[0], 
                                         format="%.6f")
            with col2:
                new_lng = st.number_input("Longitude", 
                                         value=default_coordinate[1], 
                                         format="%.6f")
            
            # Add point button
            add_point = st.form_submit_button("Add Point")
            
            if add_point:
                st.session_state.coord_input_list.append([new_lat, new_lng])
                st.rerun()
        
        # Remove last point button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Remove Last Point") and st.session_state.coord_input_list:
                st.session_state.coord_input_list.pop()
                st.rerun()
        
        with col2:
            if st.button("Clear All Points"):
                st.session_state.coord_input_list = []
                st.rerun()
        
        # Create a map to display the coordinates
        st.subheader("Preview Map")
        
        # Create a map centered at the default location or the average of coordinates
        if st.session_state.coord_input_list:
            lats = [coord[0] for coord in st.session_state.coord_input_list]
            lngs = [coord[1] for coord in st.session_state.coord_input_list]
            map_center = [(min(lats) + max(lats))/2 if lats else default_coordinate[0], 
                          (min(lngs) + max(lngs))/2 if lngs else default_coordinate[1]]
            zoom_start = 10
        else:
            map_center = st.session_state.get('map_center', default_coordinate)
            zoom_start = 12
        
        # Create a regular folium map (no drawing controls needed for preview)
        preview_map = folium.Map(location=map_center, zoom_start=zoom_start)
        
        # If coordinates are entered, draw them on the map
        if st.session_state.coord_input_list:
            # Add markers for each point
            for i, coord in enumerate(st.session_state.coord_input_list):
                folium.Marker(
                    location=coord,
                    popup=f"Point {i+1}: {coord[0]:.6f}, {coord[1]:.6f}",
                    icon=folium.Icon(icon="map-pin")
                ).add_to(preview_map)
            
            # If there are at least 2 points, connect them with a line
            if len(st.session_state.coord_input_list) >= 2:
                folium.PolyLine(
                    locations=st.session_state.coord_input_list,
                    color='blue',
                    weight=3,
                    opacity=0.7
                ).add_to(preview_map)
            
            # If there are at least 3 points, draw a polygon
            if len(st.session_state.coord_input_list) >= 3:
                # Create a closed polygon
                polygon_coords = st.session_state.coord_input_list.copy()
                # Add the first point to close the polygon
                if polygon_coords[0] != polygon_coords[-1]:
                    polygon_coords.append(polygon_coords[0])
                
                # Draw the polygon
                draw_polygon_on_map(
                    preview_map, 
                    st.session_state.coord_input_list, 
                    color='blue',
                    fill_opacity=0.2,
                    tooltip="Current Points Polygon"
                )
        
        # If an area is already selected, show it on the preview map as well
        if st.session_state.area_selected:
            # Draw boundary (either rectangle or polygon)
            if st.session_state.boundary_type == "rectangle":
                folium.Rectangle(
                    bounds=[[st.session_state.sw_corner[0], st.session_state.sw_corner[1]], 
                          [st.session_state.ne_corner[0], st.session_state.ne_corner[1]]],
                    color='red',
                    fill=True,
                    fill_opacity=0.2,
                    tooltip="Current Selected Area"
                ).add_to(preview_map)
            elif st.session_state.boundary_type == "polygon":
                # Use the custom function to draw the polygon properly
                draw_polygon_on_map(
                    preview_map, 
                    st.session_state.boundary_points, 
                    color='red',
                    fill_opacity=0.2,
                    tooltip="Current Selected Area"
                )
            
            # Add grid overlay if enabled
            if st.session_state.grid_enabled and hasattr(st.session_state, 'grid_size_km'):
                create_true_square_grid(
                    preview_map,
                    st.session_state.sw_corner,
                    st.session_state.ne_corner,
                    st.session_state.grid_size_km
                )
        
        # Display the preview map
        folium_static(preview_map, width=1000, height=500)
        
        # Set Area button - only enabled if there are at least 3 points
        if len(st.session_state.coord_input_list) >= 3:
            if st.button("Set Area from Coordinates"):
                coordinates = st.session_state.coord_input_list
                
                # Calculate bounding box from the coordinates
                lats = [coord[0] for coord in coordinates]
                lngs = [coord[1] for coord in coordinates]
                
                # Calculate SW and NE corners for the bounding rectangle
                sw_corner = [min(lats), min(lngs)]
                ne_corner = [max(lats), max(lngs)]
                
                # Update session state
                st.session_state.sw_corner = sw_corner
                st.session_state.ne_corner = ne_corner
                st.session_state.boundary_type = "polygon"
                st.session_state.boundary_points = coordinates
                st.session_state.area_selected = True
                
                # Update map center
                st.session_state.map_center = [(min(lats) + max(lats))/2, 
                                            (min(lngs) + max(lngs))/2]
                
                st.success("Area set successfully!")
                st.rerun()
        else:
            st.warning("Add at least 3 points to define an area.")
    
    # Only show drawing controls if drawing method is selected
    if selection_method == "Drawing on map":
        if st.session_state.area_selected:
            st.info("Your current selected area is shown in red. Draw a new rectangle to update it.")
        else:
            st.info("Use the rectangle tool in the top right corner of the map to select an area. Then click 'Set Drawn Area'.")
        
        # Create a map with drawing controls
        drawing_map = create_boundary_drawing_map(
            st.session_state.map_center,
            zoom_start=12,
            predefined_locations={}
        )
        
        # If an area is already selected, show it on the drawing map
        if st.session_state.area_selected:
            # Draw boundary (either rectangle or polygon)
            if st.session_state.boundary_type == "rectangle":
                folium.Rectangle(
                    bounds=[[st.session_state.sw_corner[0], st.session_state.sw_corner[1]], 
                          [st.session_state.ne_corner[0], st.session_state.ne_corner[1]]],
                    color='red',
                    fill=True,
                    fill_opacity=0.2,
                    tooltip="Current Selected Area"
                ).add_to(drawing_map)
            elif st.session_state.boundary_type == "polygon":
                # Use the custom function to draw the polygon properly
                draw_polygon_on_map(
                    drawing_map, 
                    st.session_state.boundary_points, 
                    color='red',
                    fill_opacity=0.2,
                    tooltip="Current Selected Area"
                )
            
            # Add grid overlay directly to the drawing map if enabled
            if st.session_state.grid_enabled and hasattr(st.session_state, 'grid_size_km'):
                create_true_square_grid(
                    drawing_map,
                    st.session_state.sw_corner,
                    st.session_state.ne_corner,
                    st.session_state.grid_size_km
                )
        
        # Display the map with drawing controls and capture the response
        map_data = st_folium(drawing_map, width=1000, height=600)
        
        # Process drawn features from map_data
        if map_data and 'all_drawings' in map_data and map_data['all_drawings']:
            st.session_state.last_drawn_feature = map_data['all_drawings']
            
            # Extract coordinates from the last drawing
            if len(map_data['all_drawings']) > 0:
                last_drawing = map_data['all_drawings'][-1]
                
                if 'geometry' in last_drawing and 'coordinates' in last_drawing['geometry']:
                    coords = last_drawing['geometry']['coordinates']
                    
                    # Only process rectangle shapes
                    if (last_drawing['geometry']['type'] == 'Rectangle' or 
                        (last_drawing['geometry']['type'] == 'Polygon' and len(coords[0]) == 5)):
                        
                        # For rectangles, extract the boundary points
                        if coords and coords[0]:
                            boundary_points = []
                            for coord in coords[0]:
                                # Convert [lng, lat] to [lat, lng] format
                                boundary_points.append([coord[1], coord[0]])
                            
                            # Calculate SW and NE corners
                            lats = [p[0] for p in boundary_points]
                            lngs = [p[1] for p in boundary_points]
                            
                            # Calculate SW and NE corners
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
        if st.button("Set Drawn Area", key="set_drawn_area_btn"):
            if not hasattr(st.session_state, 'last_drawn_feature') or not st.session_state.last_drawn_feature:
                st.error("Please draw a rectangle on the map first.")
            else:
                if st.session_state.boundary_type == "rectangle":
                    # Clear existing sensors and protected areas when setting a new area
                    if st.session_state.area_selected:
                        st.session_state.potential_locations = []
                        st.session_state.protected_areas = []
                        st.session_state.map_data = None
                    
                    st.session_state.area_selected = True
                    st.success("Area set successfully!")
                    st.rerun()
                else:
                    st.error("Please draw a rectangular area only.")
    
    # Button to clear drawn areas (works for both methods)
    if st.button("Clear Area", key="clear_area_btn"):
        st.session_state.area_selected = False
        st.session_state.boundary_type = "rectangle"
        st.session_state.boundary_points = []
        st.session_state.last_drawn_feature = None
        st.session_state.potential_locations = []
        st.session_state.protected_areas = []
        # Clear coordinate input list too
        if 'coord_input_list' in st.session_state:
            st.session_state.coord_input_list = []
        # Clear grid settings
        st.session_state.grid_enabled = False
        st.success("Area cleared. All sensors, protected areas, and grid settings were also cleared.")
        st.rerun()
    
    # Export options - Only show if an area is selected
    if st.session_state.area_selected:
        st.markdown("---")
        st.subheader("Export Data")
        
        # Create GeoJSON with area information
        if st.session_state.boundary_points:
            @st.cache_data
            def create_export_geojson(boundary_points, sw_corner, ne_corner, boundary_type, grid_enabled=False, grid_size_km=None):
    
                # Handle different boundary types
                if boundary_type == "rectangle":
                    # Convert boundary points from [lat, lng] to [lng, lat] for GeoJSON
                    coords = [[point[1], point[0]] for point in boundary_points]
                    # Close the polygon if needed
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                elif boundary_type == "polygon":
                    # For polygons, convert from [lat, lng] to [lng, lat] for GeoJSON
                    coords = [[point[1], point[0]] for point in boundary_points]
                    # Close the polygon if needed
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                
                # Create the GeoJSON structure
                geojson = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {
                                "type": "boundary",
                                "boundary_type": boundary_type,
                                "sw_corner": [sw_corner[0], sw_corner[1]],
                                "ne_corner": [ne_corner[0], ne_corner[1]],
                                "grid_enabled": grid_enabled,
                                "grid_size_km": grid_size_km if grid_enabled else None
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [coords]
                            }
                        }
                    ]
                }
                
                # Create the points for shapely checking (in geographic coordinates)
                # For shapely, we need [lng, lat] format for coordinates
                poly_for_check = Polygon([(point[1], point[0]) for point in boundary_points])
                
                # Add grid cells as features if grid is enabled
                if grid_enabled and grid_size_km:
                    # Earth's radius in kilometers
                    earth_radius = 6371.0  # km
                    
                    # Convert grid size from km to degrees for latitude (roughly constant)
                    # 1 degree of latitude is approximately 111 km
                    lat_deg_per_km = 1 / 111.0
                    grid_size_lat = grid_size_km * lat_deg_per_km
                    
                    # Get center latitude for longitude conversion (longitude degrees vary with latitude)
                    center_lat = (sw_corner[0] + ne_corner[0]) / 2
                    
                    # Convert grid size from km to degrees for longitude at this latitude
                    # 1 degree of longitude is approximately 111 * cos(latitude) km
                    lng_deg_per_km = 1 / (111.0 * np.cos(np.radians(center_lat)))
                    grid_size_lng = grid_size_km * lng_deg_per_km
                    
                    # Calculate the width and height of the area in kilometers
                    width_km = (ne_corner[1] - sw_corner[1]) / lng_deg_per_km
                    height_km = (ne_corner[0] - sw_corner[0]) / lat_deg_per_km
                    
                    # Calculate number of rows and columns needed
                    num_rows = int(np.ceil(height_km / grid_size_km))
                    num_cols = int(np.ceil(width_km / grid_size_km))
                    
                    # Store grid cells that are inside the polygon
                    inside_grid_cells = []
                    
                    # Process each grid cell
                    for i in range(num_rows):
                        for j in range(num_cols):
                            # Calculate the corners of this cell in degrees
                            lat_sw = sw_corner[0] + i * grid_size_lat
                            lng_sw = sw_corner[1] + j * grid_size_lng
                            
                            lat_ne = lat_sw + grid_size_lat
                            lng_ne = lng_sw + grid_size_lng
                            
                            # Calculate cell center
                            center_lat = (lat_sw + lat_ne) / 2
                            center_lng = (lng_sw + lng_ne) / 2
                            
                            # Check if center is inside the polygon
                            center_point = Point(center_lng, center_lat)
                            is_inside = poly_for_check.contains(center_point)
                            
                            # Only add grid cells that are inside the polygon
                            if is_inside:
                                # GeoJSON coordinates for this cell (in [lng, lat] format)
                                cell_coords = [
                                    [lng_sw, lat_sw],  # SW
                                    [lng_ne, lat_sw],  # SE
                                    [lng_ne, lat_ne],  # NE
                                    [lng_sw, lat_ne],  # NW
                                    [lng_sw, lat_sw]   # Close the polygon
                                ]
                                
                                # Create feature for this grid cell
                                grid_feature = {
                                    "type": "Feature",
                                    "properties": {
                                        "type": "grid_cell",
                                        "grid_id": f"r{i}c{j}",
                                        "row": i,
                                        "col": j,
                                        "sw_corner": [lat_sw, lng_sw],  # in [lat, lng] format for easier use
                                        "ne_corner": [lat_ne, lng_ne],  # in [lat, lng] format for easier use
                                        "center": [center_lat, center_lng],  # in [lat, lng] format
                                        "size_km": grid_size_km
                                    },
                                    "geometry": {
                                        "type": "Polygon",
                                        "coordinates": [cell_coords]
                                    }
                                }
                                
                                # Add grid cell to the features list
                                geojson["features"].append(grid_feature)
                                
                                # Also store in our list for summary
                                inside_grid_cells.append({
                                    "grid_id": f"r{i}c{j}",
                                    "center": [center_lat, center_lng],
                                    "corners": {
                                        "sw": [lat_sw, lng_sw],
                                        "ne": [lat_ne, lng_ne],
                                        "se": [lat_sw, lng_ne],
                                        "nw": [lat_ne, lng_sw]
                                    }
                                })
                    
                    # Add a summary feature with just the count of inside cells
                    summary_feature = {
                        "type": "Feature",
                        "properties": {
                            "type": "grid_summary",
                            "total_cells_inside": len(inside_grid_cells),
                            "total_rows": num_rows,
                            "total_cols": num_cols,
                            "total_cells": num_rows * num_cols
                        },
                        "geometry": None
                    }
                    
                    geojson["features"].append(summary_feature)
                
                return json.dumps(geojson, indent=2)
            # Get grid parameters from session state
            grid_enabled = st.session_state.get('grid_enabled', False)
            grid_size_km = st.session_state.get('grid_size_km', 1.0) if grid_enabled else None
            
            # Create exportable GeoJSON with area and grid data
            export_geojson = create_export_geojson(
                st.session_state.boundary_points,
                st.session_state.sw_corner,
                st.session_state.ne_corner,
                st.session_state.boundary_type,
                grid_enabled,
                grid_size_km
            )
            
            # Download button
            st.download_button(
                label="Download Area and Grid as GeoJSON",
                data=export_geojson,
                file_name='selected_area_with_grid.geojson',
                mime='application/json',
            )