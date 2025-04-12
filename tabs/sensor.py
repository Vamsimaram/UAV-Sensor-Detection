# tabs/sensor.py
import streamlit as st
import folium
from streamlit_folium import folium_static, st_folium
import json
from datetime import datetime
from folium.plugins import Draw

# Import from map_utils
from map_utils import is_point_in_rectangle, is_point_in_polygon

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