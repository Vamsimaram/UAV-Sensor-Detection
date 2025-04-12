# tabs/protected_areas.py
import streamlit as st
import folium
from streamlit_folium import folium_static, st_folium
import json
from datetime import datetime
from folium.plugins import Draw

# Import from map_utils
from map_utils import is_point_in_rectangle, is_point_in_polygon

def protected_areas_tab():
    """
    Map-based polygon drawing for protected areas within the selected region
    """
    st.header("Protected Areas")
    
    if not st.session_state.area_selected:
        st.warning("Please select an area of interest in the Map & Selection tab first.")
        return
    
    # Initialize session state for protected areas if not already done
    if 'protected_areas' not in st.session_state:
        st.session_state.protected_areas = []
    
    # Create columns for layout - using different ratio for better layout
    map_col, list_col = st.columns([4, 1])
    
    # Display list of protected areas in the right column
    with list_col:
        st.subheader("Protected Areas")
        
        # Show total count
        if st.session_state.protected_areas:
            st.info(f"Total areas: {len(st.session_state.protected_areas)}")
            
            # Create a scrollable container for the areas list
            areas_container = st.container()
            with areas_container:
                # Display each protected area with a remove button
                for i, area in enumerate(st.session_state.protected_areas):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**Area {i+1}:** {len(area['points'])} points")
                    with col2:
                        # Add a remove button for each area
                        if st.button("🗑️", key=f"remove_area_{i}"):
                            st.session_state.protected_areas.pop(i)
                            # Clear map data to prevent issues with area indexing
                            st.session_state.protected_area_map_data = None
                            st.rerun()
            
            # Export and Clear buttons for protected areas
            if st.session_state.protected_areas:
                st.markdown("---")
                
                # Create the GeoJSON data
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": []
                }
                
                for area in st.session_state.protected_areas:
                    # Convert points from [lat, lng] to [lng, lat] for GeoJSON
                    coordinates = [[point[1], point[0]] for point in area['points']]
                    # Close the polygon by adding the first point at the end if needed
                    if coordinates[0] != coordinates[-1]:
                        coordinates.append(coordinates[0])
                    
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "name": area.get('name', f"Protected Area {len(geojson_data['features'])+1}"),
                            "timestamp": area.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [coordinates]
                        }
                    }
                    geojson_data["features"].append(feature)
                
                # Convert to JSON string
                geojson_str = json.dumps(geojson_data, indent=2)
                
                # Export button
                st.download_button(
                    label="📋 Export GeoJSON",
                    data=geojson_str,
                    file_name='protected_areas.geojson',
                    mime='application/geo+json',
                    key="export_protected_areas_button"
                )
                
                # Clear button
                if st.button("🗑️ Clear All", key="clear_protected_areas_button"):
                    st.session_state.protected_areas = []
                    st.session_state.protected_area_map_data = None
                    st.rerun()
        else:
            st.info("No protected areas defined yet.")
            st.markdown("""
            **Instructions:**
            1. Click the polygon tool in the top right corner of the map
            2. Draw a polygon within the red highlighted area
            3. Click "Set Protected Areas" after drawing EACH polygon
            4. Export the areas when finished
            """)
    
    with map_col:
        st.subheader("Draw Protected Areas")
        st.warning("IMPORTANT: Draw ONE polygon at a time and click 'Set Protected Areas' after each drawing. All points of the polygon must be inside the selected region.")
        
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
        
        # Display all previously defined protected areas with green color
        for i, area in enumerate(st.session_state.protected_areas):
            folium.Polygon(
                locations=area['points'],
                color='green',
                weight=2,
                fill=True,
                fill_color='green',
                fill_opacity=0.3,
                popup=f"Protected Area {i+1}"
            ).add_to(m)
            
        # Display all sensors from the sensor placement tab with blue markers
        if 'potential_locations' in st.session_state:
            for i, location in enumerate(st.session_state.potential_locations):
                folium.Marker(
                    location=[location['lat'], location['lng']],
                    popup=f"Sensor {i+1}: [{location['lat']:.6f}, {location['lng']:.6f}]",
                    tooltip=f"Sensor {i+1}",
                    icon=folium.Icon(color='blue', icon='circle', prefix='fa')
                ).add_to(m)
        
        # Add the Draw control to the map - only enable polygon
        draw = Draw(
            export=False,  # Disable export to prevent duplications
            position='topright',
            draw_options={
                'polyline': False,
                'polygon': True,
                'rectangle': False,
                'circle': False,
                'circlemarker': False,
                'marker': False,
            },
            edit_options={
                'featureGroup': None,
                'remove': True  # Allow removal directly from the map
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
            <h4 style="margin-top: 0;">Protected Area Instructions:</h4>
            <ul style="padding-left: 20px; margin-bottom: 0;">
                <li>Click the polygon tool in the top right corner</li>
                <li>Draw a polygon to define a protected area</li>
                <li>Click "Set Protected Areas" after EACH polygon</li>
                <li>Draw multiple polygons for separate protected areas</li>
                <li>Use the Export button to save all protected areas</li>
            </ul>
            <p style="color: red; font-weight: bold; margin-top: 5px; margin-bottom: 0;">
                Note: All points of the polygon must be inside the highlighted region!
            </p>
            <p style="margin-top: 5px; margin-bottom: 0;">
                <b>Blue circles</b>: Sensor locations from Sensor tab
            </p>
        </div>
        """
        instructions = folium.Element(instructions_html)
        m.get_root().html.add_child(instructions)
        
        # Display the map and get drawing data
        map_data = st_folium(m, width=800, height=600, key="protected_areas_map_display")
        
        # Store the map data in session state to prevent it from disappearing on rerun
        if map_data:
            st.session_state.protected_area_map_data = map_data
        
        # Initialize the session state for map data if not already done
        if 'protected_area_map_data' not in st.session_state:
            st.session_state.protected_area_map_data = None
        
        # Automatically generate area name - no user input required
        
        # Button to set drawn protected areas
        if st.button("Set Protected Areas", key="set_protected_areas_button"):
            if (st.session_state.protected_area_map_data and 
                'all_drawings' in st.session_state.protected_area_map_data and 
                st.session_state.protected_area_map_data['all_drawings']):
                
                new_areas = []
                invalid_areas = 0
                
                for feature in st.session_state.protected_area_map_data['all_drawings']:
                    if feature['geometry']['type'] == 'Polygon':
                        # Extract coordinates from the drawing
                        coords = feature['geometry']['coordinates']
                        
                        # For polygons, coordinates are in the format [[[lng1, lat1], [lng2, lat2], ...]]
                        if coords and coords[0]:
                            # Convert [lng, lat] to [lat, lng] format for our application
                            points = []
                            for coord in coords[0]:
                                points.append([coord[1], coord[0]])
                            
                            # Check if ALL points of the polygon are inside the selected area
                            is_valid = True  # Start with True and set to False if any point is outside
                            for point in points:
                                if st.session_state.boundary_type == "rectangle":
                                    if not is_point_in_rectangle(
                                        point, 
                                        st.session_state.sw_corner, 
                                        st.session_state.ne_corner
                                    ):
                                        is_valid = False
                                        break
                                elif st.session_state.boundary_type == "polygon":
                                    if not is_point_in_polygon(
                                        point, 
                                        st.session_state.boundary_points
                                    ):
                                        is_valid = False
                                        break
                            
                            if is_valid:
                                # Create a new protected area with automatic numbering
                                new_area = {
                                    'points': points,
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'name': f"Protected Area {len(st.session_state.protected_areas) + 1}"
                                }
                                new_areas.append(new_area)
                            else:
                                invalid_areas += 1
                
                # Add new protected areas to the session state
                if new_areas:
                    st.session_state.protected_areas.extend(new_areas)
                    
                    if invalid_areas > 0:
                        st.warning(f"Added {len(new_areas)} new protected area(s). {invalid_areas} area(s) had points outside the selected region and were ignored.")
                    else:
                        st.success(f"Added {len(new_areas)} new protected area(s)!")
                    
                    # Clear the map_data after adding areas to allow for new areas to be drawn
                    st.session_state.protected_area_map_data = None
                    
                    st.rerun()
                else:
                    if invalid_areas > 0:
                        st.warning(f"No areas were added. All {invalid_areas} area(s) had points outside the selected region and were ignored.")
                    else:
                        st.info("No new valid protected areas were drawn.")
            else:
                st.warning("No polygons detected. Please draw at least one polygon on the map first.")