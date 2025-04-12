import streamlit as st
import folium
from streamlit_folium import folium_static, st_folium
import pandas as pd
import json

# Import from map_utils
from map_utils import create_boundary_drawing_map

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
            # Clear existing sensors and protected areas when setting a new area
            if st.session_state.area_selected:
                # This is a change of existing area, clear sensors and protected areas
                st.session_state.potential_locations = []
                st.session_state.protected_areas = []
                st.session_state.map_data = None
            
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
            # st.subheader("Your Selected Area")
            # folium_static(selected_area_map)
            
            # st.success("Drawn area set successfully! The highlighted area will appear in all tabs.")
    
    if st.button("Clear Drawn Areas"):
        st.session_state.area_selected = False
        st.session_state.boundary_type = "rectangle"
        st.session_state.boundary_points = []
        st.session_state.last_drawn_feature = None
        st.session_state.potential_locations = []  # Clear sensors
        st.session_state.protected_areas = []      # Clear protected areas
        st.success("Drawn areas cleared. All sensors and protected areas were also cleared.")
    
    # Grid size input (we'll keep this since it's needed for some functionality)
    grid_size = st.slider("Grid Size (degrees)", min_value=0.001, max_value=0.05, value=0.01, step=0.001)
    st.session_state.grid_size = grid_size
    
    # Display the current boundary points if available and add download option
    if st.session_state.boundary_points:
        st.subheader("Current Boundary Points")
        points_df = pd.DataFrame(st.session_state.boundary_points, columns=["Latitude", "Longitude"])
        st.dataframe(points_df)
        
        # Add option to download as GeoJSON only
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