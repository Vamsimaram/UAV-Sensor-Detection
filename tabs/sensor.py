import streamlit as st
import folium
from streamlit_folium import folium_static, st_folium
import json
import re
from datetime import datetime
from folium.plugins import Draw

# Import from map_utils
from map_utils import is_point_in_rectangle, is_point_in_polygon

def sensor_tab():
    """
    Map-based point selection for potential sensor locations using the Draw plugin
    """
    st.header("Potential Sensor Locations")
    
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
    
    # Check if protected areas have been defined
    if not st.session_state.protected_areas:
        st.warning("Please define protected areas in the Protected Areas tab first before placing sensors.")
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
                        # Display sensor name if it exists, otherwise show coordinates
                        display_text = f"**{location.get('name', f'Sensor {i+1}')}:** [{location['lat']:.6f}, {location['lng']:.6f}]"
                        st.write(display_text)
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
                    "type": "Sensor Point Collection",
                    "features": []
                }

                for location in st.session_state.potential_locations:
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "name": location.get('name', f"Sensor {i+1}")  # Add the sensor name to properties
                        },
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
            3. Enter a unique name (cannot start with a number) 
            4. Click "Set Drawn Sensors" after placing EACH marker
            5. Export the points when finished
            """)
    
    with map_col:
        st.subheader("Mark Potential Sensor Locations")
        st.warning("IMPORTANT: Place ONE point at a time and when you select all points, click 'Set Drawn Sensors'. Points outside the highlighted area will be ignored.")
        
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
        
        # Display all protected areas with green color
        for i, area in enumerate(st.session_state.protected_areas):
            folium.Polygon(
                locations=area['points'],
                color='green',
                weight=2,
                fill=True,
                fill_color='green',
                fill_opacity=0.3,
                tooltip=area.get('name', f"Protected Area {i+1}"),  # Show area name on hover
                popup=f"{area.get('name', f'Protected Area {i+1}')}"
            ).add_to(m)
        
        # Display all marked potential locations with highly visible markers
        for i, location in enumerate(st.session_state.potential_locations):
            # Use the sensor name if it exists, otherwise use "Sensor X"
            sensor_name = location.get('name', f"Sensor {i+1}")
            
            folium.Marker(
                location=[location['lat'], location['lng']],
                popup=f"{sensor_name}: [{location['lat']:.6f}, {location['lng']:.6f}]",
                tooltip=sensor_name,  # Show name on hover
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
                <li>Enter a unique name (cannot start with a number)</li>
                <li>Click "Set Drawn Sensors" after EACH point</li>
                <li>Points must be inside the red highlighted area</li>
                <li>Hover over sensors to see their names</li>
                <li>Use the Export button to save your sensor locations</li>
            </ul>
            <p style="color: red; font-weight: bold; margin-top: 5px; margin-bottom: 0;">
                Note: Points outside the highlighted area will be ignored!
            </p>
            <p style="margin-top: 5px; margin-bottom: 0;">
                <b>Green areas</b>: Protected areas (sensors can be placed inside)
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
            
        # Initialize and manage the sensor name input
        if 'sensor_name_key' not in st.session_state:
            st.session_state.sensor_name_key = 0
        
        # Generate a unique key for the text input that changes when we want to clear it
        input_key = f"sensor_name_input_{st.session_state.sensor_name_key}"
        
        # Add an input field for the sensor name
        sensor_name = st.text_input("Enter Sensor Name:", 
                               key=input_key,
                               placeholder="Name cannot start with a number and must be unique")
        
        # Button to set drawn sensors
        if st.button("Set Drawn Sensors", key="set_sensors_button"):
            # Validate the name first
            valid_name = True
            error_message = ""
            
            # Check if name is provided
            if not sensor_name:
                valid_name = False
                error_message = "Please enter a name for the sensor."
            # Check if name starts with a number
            elif re.match(r'^\d', sensor_name):
                valid_name = False
                error_message = "Sensor name cannot start with a number."
            # Check for duplicate names
            elif any(location.get('name') == sensor_name for location in st.session_state.potential_locations):
                valid_name = False
                error_message = f"The name '{sensor_name}' is already in use. Please choose a unique name."
            
            if not valid_name:
                st.error(error_message)
            elif st.session_state.map_data and 'all_drawings' in st.session_state.map_data and st.session_state.map_data['all_drawings']:
                new_sensors = []
                invalid_points = 0
                existing_locations = {(loc['lat'], loc['lng']) for loc in st.session_state.potential_locations}
                
                for feature in st.session_state.map_data['all_drawings']:
                    if feature['geometry']['type'] == 'Point':
                        lng, lat = feature['geometry']['coordinates']
                        
                        # Check if the point is inside the selected area
                        is_inside_boundary = False
                        if st.session_state.boundary_type == "rectangle":
                            is_inside_boundary = is_point_in_rectangle(
                                [lat, lng], 
                                st.session_state.sw_corner, 
                                st.session_state.ne_corner
                            )
                        elif st.session_state.boundary_type == "polygon":
                            is_inside_boundary = is_point_in_polygon(
                                [lat, lng], 
                                st.session_state.boundary_points
                            )
                        
                        # Only check if point is inside the boundary - no longer checking protected areas
                        if is_inside_boundary and (lat, lng) not in existing_locations:
                            new_sensor = {
                                'lat': lat,
                                'lng': lng,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'name': sensor_name
                            }
                            new_sensors.append(new_sensor)
                            existing_locations.add((lat, lng))
                        else:
                            if not is_inside_boundary:
                                invalid_points += 1
                
                # Add new sensors to the potential locations
                if new_sensors:
                    st.session_state.potential_locations.extend(new_sensors)
                    
                    message_parts = []
                    if new_sensors:
                        message_parts.append(f"Added '{sensor_name}' as a new sensor location.")
                    if invalid_points > 0:
                        message_parts.append(f"{invalid_points} point(s) were outside the selected region.")
                    
                    if invalid_points > 0:
                        st.warning(" ".join(message_parts))
                    else:
                        st.success(" ".join(message_parts))
                    
                    # Important: Clear the map_data after adding points to allow for new points to be added
                    st.session_state.map_data = None
                    
                    # Increment the key to force a new text input with an empty value
                    st.session_state.sensor_name_key += 1
                    
                    st.rerun()
                else:
                    message_parts = []
                    message_parts.append("No points were added.")
                    
                    if invalid_points > 0:
                        message_parts.append(f"{invalid_points} point(s) were outside the selected region.")
                    
                    st.warning(" ".join(message_parts))
            else:
                st.warning("No markers detected. Please place markers on the map first.")