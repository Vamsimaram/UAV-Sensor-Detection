import streamlit as st
import folium
from streamlit_folium import folium_static, st_folium
import pandas as pd
import json
import math

# Import from map_utils
from map_utils import create_boundary_drawing_map, create_grid_overlay

def map_selection_tab():
    """
    Map-based selection tab for defining area of interest
    """
    st.header("Select Area of Interest")
    
    st.subheader("Draw Boundary on Map")
    
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
        # Draw rectangle boundary
        folium.Rectangle(
            bounds=[[st.session_state.sw_corner[0], st.session_state.sw_corner[1]], 
                  [st.session_state.ne_corner[0], st.session_state.ne_corner[1]]],
            color='red',
            fill=True,
            fill_opacity=0.2,
            tooltip="Current Selected Area"
        ).add_to(drawing_map)
        
        # Add grid overlay if enabled
        if st.session_state.get('show_grid', False):
            create_grid_overlay(
                drawing_map, 
                st.session_state.sw_corner, 
                st.session_state.ne_corner, 
                st.session_state.grid_size
            )
    
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
                st.success("Area set successfully! You can now configure the grid below.")
            else:
                st.error("Please draw a rectangular area only.")
    
    # Button to clear drawn areas
    if st.button("Clear Area", key="clear_area_btn"):
        st.session_state.area_selected = False
        st.session_state.boundary_type = "rectangle"
        st.session_state.boundary_points = []
        st.session_state.last_drawn_feature = None
        st.session_state.potential_locations = []
        st.session_state.protected_areas = []
        st.success("Area cleared. All sensors and protected areas were also cleared.")
    
    # Only show grid configuration if an area is selected
    if st.session_state.area_selected:
        st.markdown("---")
        st.subheader("Grid Configuration")
        
        # Calculate area dimensions in meters for reference
        lat_diff = st.session_state.ne_corner[0] - st.session_state.sw_corner[0]
        lng_diff = st.session_state.ne_corner[1] - st.session_state.sw_corner[1]
        
        # Convert to meters (approximate)
        avg_lat = (st.session_state.ne_corner[0] + st.session_state.sw_corner[0]) / 2
        lat_meters = lat_diff * 111000
        lng_meters = lng_diff * 111000 * math.cos(math.radians(avg_lat))
        
        st.info(f"Selected area dimensions: {lat_meters:.1f}m × {lng_meters:.1f}m")
        
        # Grid unit selection: meters or kilometers
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Let user specify grid cell size in real-world units
            grid_size_unit = st.selectbox("Grid Cell Unit", 
                                        options=["meters", "kilometers"],
                                        index=0)
        
        with col2:
            # Input for grid cell size
            if grid_size_unit == "meters":
                min_val = 10  # Minimum 10 meters
                max_val = min(10000, min(lat_meters, lng_meters))  # Max 10km or area size
                default_val = min(100, max_val/2)  # Default 100m or half of max
                step_val = 10  # Step by 10 meters
                
                # Change this line in map_selection.py (line 153)
                grid_cell_size = st.number_input("Grid Cell Size (meters)", 
                                                value=100.0,      # Float
                                                min_value=10.0,   # Float
                                                max_value=1000.0, # Float
                                                step=10.0)        # Float
                # Convert to degrees
                grid_size_degrees = grid_cell_size / (111000 * math.cos(math.radians(avg_lat)))
            else:  # kilometers
                min_val = 0.01  # Minimum 10 meters (0.01 km)
                max_val = min(10, min(lat_meters, lng_meters)/1000)  # Max 10km or area size
                default_val = min(1, max_val/2)  # Default 1km or half of max
                step_val = 0.1  # Step by 100 meters (0.1 km)
                
                grid_cell_size = st.number_input("Grid Cell Size (kilometers)", 
                                                min_value=min_val,
                                                max_value=max_val,
                                                value=float(default_val),
                                                step=step_val)
                # Convert to degrees - remember to convert km to meters first
                grid_size_degrees = (grid_cell_size * 1000) / (111000 * math.cos(math.radians(avg_lat)))
        
        with col3:
            # Toggle for showing/hiding grid
            show_grid = st.toggle("Show Grid", value=st.session_state.get('show_grid', False))
            st.session_state.show_grid = show_grid
        
        # Update grid size in session state
        st.session_state.grid_size = grid_size_degrees
        
        # Button to update the grid
        if st.button("Update Grid", key="update_grid_btn"):
            # Calculate grid dimensions
            if st.session_state.show_grid:
                lat_cells = int(lat_diff / grid_size_degrees)
                lng_cells = int(lng_diff / grid_size_degrees)
                cell_size_display = f"{grid_cell_size} {grid_size_unit}"
                
                st.success(f"Grid updated: {lat_cells} × {lng_cells} cells, each {cell_size_display}")
            else:
                st.info("Grid is currently hidden. Toggle 'Show Grid' to display it.")
            
            st.rerun()
        
        # Display the area with grid
        if st.session_state.area_selected:
            st.subheader("Selected Area Preview")
            
            preview_map = folium.Map(
                location=st.session_state.map_center, 
                zoom_start=12
            )
            
            # Draw rectangle boundary
            folium.Rectangle(
                bounds=[[st.session_state.sw_corner[0], st.session_state.sw_corner[1]], 
                    [st.session_state.ne_corner[0], st.session_state.ne_corner[1]]],
                color='red',
                fill=True,
                fill_opacity=0.3,
                tooltip="Selected Area"
            ).add_to(preview_map)
            
            # Add grid overlay if enabled
            if st.session_state.show_grid:
                lat_cells = int(lat_diff / grid_size_degrees)
                lng_cells = int(lng_diff / grid_size_degrees)
                
                # Get adjusted grid size to ensure exact fit
                adjusted_lat_grid_size = lat_diff / lat_cells if lat_cells > 0 else grid_size_degrees
                adjusted_lng_grid_size = lng_diff / lng_cells if lng_cells > 0 else grid_size_degrees
                
                # Create grid with adjusted cell size
                create_grid_overlay(
                    preview_map, 
                    st.session_state.sw_corner, 
                    st.session_state.ne_corner, 
                    [adjusted_lat_grid_size, adjusted_lng_grid_size]  # Pass as list for separate lat/lng sizes
                )
                
                # Calculate grid stats
                cell_lat_meters = adjusted_lat_grid_size * 111000
                cell_lng_meters = adjusted_lng_grid_size * 111000 * math.cos(math.radians(avg_lat))
                
                st.info(f"Grid dimensions: {lat_cells} rows × {lng_cells} columns = {lat_cells * lng_cells} cells total")
                st.info(f"Actual cell size: {cell_lat_meters:.1f}m × {cell_lng_meters:.1f}m")
            
            # Display the map with the selected area and grid
            folium_static(preview_map)
        
        # Export options
        st.markdown("---")
        st.subheader("Export Data")
        
        # Create GeoJSON with grid information
        if st.session_state.boundary_points:
            @st.cache_data
            def create_export_geojson(boundary_points, sw_corner, ne_corner, grid_size, show_grid, grid_unit, grid_cell_size):
                # Convert boundary points from [lat, lng] to [lng, lat] for GeoJSON
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
                                "sw_corner": [sw_corner[0], sw_corner[1]],
                                "ne_corner": [ne_corner[0], ne_corner[1]]
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [coords]
                            }
                        }
                    ],
                    "grid_metadata": {
                        "enabled": show_grid,
                        "cell_size_degrees": grid_size,
                        "cell_size_value": grid_cell_size,
                        "cell_size_unit": grid_unit
                    }
                }
                
                # Add grid cells as features if grid is enabled
                if show_grid:
                    # Calculate lat and lng differences
                    lat_diff = ne_corner[0] - sw_corner[0]
                    lng_diff = ne_corner[1] - sw_corner[1]
                    
                    # Calculate number of cells
                    lat_cells = int(lat_diff / grid_size)
                    lng_cells = int(lng_diff / grid_size)
                    
                    # Adjust grid size to fit exactly
                    adjusted_lat_grid_size = lat_diff / lat_cells if lat_cells > 0 else grid_size
                    adjusted_lng_grid_size = lng_diff / lng_cells if lng_cells > 0 else grid_size
                    
                    # Create grid cells
                    for i in range(lat_cells):
                        for j in range(lng_cells):
                            lat_sw = sw_corner[0] + i * adjusted_lat_grid_size
                            lng_sw = sw_corner[1] + j * adjusted_lng_grid_size
                            
                            lat_ne = lat_sw + adjusted_lat_grid_size
                            lng_ne = lng_sw + adjusted_lng_grid_size
                            
                            # Create cell coordinates
                            cell_coords = [
                                [lng_sw, lat_sw],  # SW
                                [lng_ne, lat_sw],  # SE
                                [lng_ne, lat_ne],  # NE
                                [lng_sw, lat_ne],  # NW
                                [lng_sw, lat_sw]   # SW again to close
                            ]
                            
                            # Create cell feature
                            cell_feature = {
                                "type": "Feature",
                                "properties": {
                                    "type": "grid_cell",
                                    "row": i,
                                    "col": j,
                                    "cell_id": f"r{i}c{j}"
                                },
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [cell_coords]
                                }
                            }
                            
                            geojson["features"].append(cell_feature)
                
                return json.dumps(geojson, indent=2)
            
            # Create exportable GeoJSON with all data
            export_geojson = create_export_geojson(
                st.session_state.boundary_points,
                st.session_state.sw_corner,
                st.session_state.ne_corner,
                st.session_state.grid_size,
                st.session_state.show_grid,
                grid_size_unit,
                grid_cell_size
            )
            
            # Download button
            st.download_button(
                label="Download Area with Grid as GeoJSON",
                data=export_geojson,
                file_name='area_with_grid.geojson',
                mime='application/json',
            )
    else:
        # Show message if no area is selected
        st.info("Please select an area of interest by drawing on the map above to configure the grid.")