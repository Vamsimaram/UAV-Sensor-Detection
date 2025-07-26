# tabs/protected_areas.py
import streamlit as st
import folium
from streamlit_folium import folium_static, st_folium
import json
import re
from datetime import datetime
from folium.plugins import Draw

# Import from map_utils
from map_utils import is_point_in_rectangle, is_point_in_polygon

def protected_areas_tab():
    """
    Map-based polygon drawing for protected areas within the selected region
    """
    st.header("Protected Areas")
    
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
                        # Show only unique points count (not duplicate first point)
                        unique_points = len(area['points'])
                        st.write(f"**{area['name']}:** {unique_points} points")
                        # Add expander to show coordinates
                        with st.expander("View Coordinates"):
                            for j, point in enumerate(area['points']):
                                st.write(f"Point {j+1}: [{point[0]:.6f}, {point[1]:.6f}]")
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
            3. Enter a unique name (cannot start with a number)
            4. Click "Set Protected Areas" after drawing EACH polygon
            5. Export the areas when finished
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
            # We need to close the polygon for display, but not for storage
            display_points = area['points'].copy()
            if display_points[0] != display_points[-1]:
                display_points.append(display_points[0])
                
            folium.Polygon(
                locations=display_points,
                color='green',
                weight=2,
                fill=True,
                fill_color='green',
                fill_opacity=0.3,
                tooltip=area['name'],  # Add tooltip to show area name on hover
                popup=f"{area['name']}: {len(area['points'])} points"
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
        
        # Display the map and get drawing data
        map_data = st_folium(m, width=800, height=600, key="protected_areas_map_display")
        
        # Store the map data in session state to prevent it from disappearing on rerun
        if map_data:
            st.session_state.protected_area_map_data = map_data
        
        # Initialize the session state for map data if not already done
        if 'protected_area_map_data' not in st.session_state:
            st.session_state.protected_area_map_data = None
        
        # Initialize and manage the area name input
        if 'should_clear_name' not in st.session_state:
            st.session_state.should_clear_name = False
        
        # Generate a unique key for the text input that changes when we want to clear it
        if 'name_input_key' not in st.session_state:
            st.session_state.name_input_key = 0
        
        # When we want to clear, we'll increment this key
        input_key = f"protected_area_name_input_{st.session_state.name_input_key}"
        
        # Add an input field for the area name
        area_name = st.text_input("Enter Protected Area Name:", 
                                key=input_key,
                                placeholder="Name cannot start with a number and must be unique")
        
        # Button to set drawn protected areas
        if st.button("Set Protected Areas", key="set_protected_areas_button"):
            # Validate the name first
            valid_name = True
            error_message = ""
            
            # Check if name is provided
            if not area_name:
                valid_name = False
                error_message = "Please enter a name for the protected area."
            # Check if name starts with a number
            elif re.match(r'^\d', area_name):
                valid_name = False
                error_message = "Protected area name cannot start with a number."
            # Check for duplicate names
            elif any(area['name'] == area_name for area in st.session_state.protected_areas):
                valid_name = False
                error_message = f"The name '{area_name}' is already in use. Please choose a unique name."
            
            if not valid_name:
                st.error(error_message)
            elif (st.session_state.protected_area_map_data and 
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
                            # Remove the duplicated first point if it exists (last == first)
                            points = []
                            raw_points = coords[0]
                            # Check if the last point is a duplicate of the first point
                            if raw_points[0][0] == raw_points[-1][0] and raw_points[0][1] == raw_points[-1][1]:
                                # Remove the last point as it's a duplicate
                                raw_points = raw_points[:-1]
                                
                            for coord in raw_points:
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
                                # Create a new protected area with the user provided name
                                new_area = {
                                    'points': points,  # Only store unique points
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'name': area_name
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
                        st.success(f"Added '{area_name}' as a new protected area with {len(new_areas[0]['points'])} unique points!")
                    
                    # Clear the map_data after adding areas to allow for new areas to be drawn
                    st.session_state.protected_area_map_data = None
                    
                    # Increment the key to force a new text input with an empty value
                    st.session_state.name_input_key += 1
                    
                    st.rerun()
                else:
                    if invalid_areas > 0:
                        st.warning(f"No areas were added. All {invalid_areas} area(s) had points outside the selected region and were ignored.")
                    else:
                        st.info("No new valid protected areas were drawn.")
            else:
                st.warning("No polygons detected. Please draw at least one polygon on the map first.")