# prediction_results_visualizer.py
import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

def create_all_grid_cells_with_inside_info(sw_corner, ne_corner, grid_size_degrees):
    """
    Create ALL grid cells (like map_selection.py) but mark which ones are inside the boundary
    This shows the complete grid but only allows coloring of inside cells
    """
    # Get center latitude for longitude adjustment
    center_lat = (sw_corner[0] + ne_corner[0]) / 2
    
    # Adjust longitude grid size to make cells square
    lat_grid_size = grid_size_degrees
    lng_grid_size = grid_size_degrees / np.cos(np.radians(center_lat))
    
    # Calculate the width and height of the area
    width_degrees = ne_corner[1] - sw_corner[1]
    height_degrees = ne_corner[0] - sw_corner[0]
    
    # Calculate number of rows and columns needed to cover the area
    num_rows = int(np.ceil(height_degrees / lat_grid_size))
    num_cols = int(np.ceil(width_degrees / lng_grid_size))
    
    all_grid_cells = []
    inside_cells_count = 0
    
    # Debug info
    print(f"DEBUG: SW Corner: {sw_corner}, NE Corner: {ne_corner}")
    print(f"DEBUG: Grid size: {grid_size_degrees} degrees")
    print(f"DEBUG: Lat grid size: {lat_grid_size}, Lng grid size: {lng_grid_size}")
    print(f"DEBUG: Num rows: {num_rows}, Num cols: {num_cols}")
    print(f"DEBUG: Boundary type: {st.session_state.boundary_type}")
    
    for i in range(num_rows):
        for j in range(num_cols):
            # Calculate the corners of this cell
            lat_sw = sw_corner[0] + i * lat_grid_size
            lng_sw = sw_corner[1] + j * lng_grid_size
            
            lat_ne = lat_sw + lat_grid_size
            lng_ne = lng_sw + lng_grid_size
            
            # Calculate cell center
            center_lat_cell = (lat_sw + lat_ne) / 2
            center_lng_cell = (lng_sw + lng_ne) / 2
            
            # Check if center is inside the boundary
            if st.session_state.boundary_type == "rectangle":
                is_inside = (sw_corner[0] <= center_lat_cell <= ne_corner[0] and 
                            sw_corner[1] <= center_lng_cell <= ne_corner[1])
                # Debug first few cells
                if i < 3 and j < 3:
                    print(f"DEBUG: Cell ({i},{j}) center: [{center_lat_cell:.6f}, {center_lng_cell:.6f}], inside: {is_inside}")
            elif st.session_state.boundary_type == "polygon":
                is_inside = is_point_in_polygon(
                    [center_lat_cell, center_lng_cell],
                    st.session_state.boundary_points
                )
                # Debug first few cells
                if i < 3 and j < 3:
                    print(f"DEBUG: Cell ({i},{j}) center: [{center_lat_cell:.6f}, {center_lng_cell:.6f}], inside: {is_inside}")
                    print(f"DEBUG: Boundary points: {st.session_state.boundary_points[:3]}...")
            else:
                is_inside = True
            
            # Create grid cell info
            grid_cell = {
                'bounds': [[lat_sw, lng_sw], [lat_ne, lng_ne]],
                'row': i,
                'col': j,
                'center': [center_lat_cell, center_lng_cell],
                'lat_size': lat_grid_size,
                'lng_size': lng_grid_size,
                'is_inside': is_inside,
                'pd_array_index': inside_cells_count if is_inside else None  # Only inside cells get Pd values
            }
            
            all_grid_cells.append(grid_cell)
            
            if is_inside:
                inside_cells_count += 1
    
    print(f"DEBUG: Total cells: {len(all_grid_cells)}, Inside cells: {inside_cells_count}")
    return all_grid_cells, num_rows, num_cols, inside_cells_count

def create_prediction_results_map(model_results, target_type, selection_level="all_sensors", selected_option="for_all_sensors", classification_type="target_detection"):
    """
    Create a map showing prediction results with colored grid cells based on Pd values
    
    Parameters:
    -----------
    model_results : dict
        The output from the analytical model
    target_type : str
        The selected target type (UAV configuration)
    selection_level : str
        "all_sensors", "by_type", or "by_individual" 
    selected_option : str
        The specific option selected
    classification_type : str
        Either "target_detection" or "foe_or_friend"
    """
    
    # Create base map
    m = folium.Map(location=st.session_state.map_center, zoom_start=14)
    
    # Draw the selected area boundary
    if st.session_state.boundary_type == "rectangle":
        folium.Rectangle(
            bounds=[[st.session_state.sw_corner[0], st.session_state.sw_corner[1]], 
                   [st.session_state.ne_corner[0], st.session_state.ne_corner[1]]],
            color='red',
            weight=2,
            fill=False,
            popup="Selected Area"
        ).add_to(m)
    elif st.session_state.boundary_type == "polygon":
        folium.Polygon(
            locations=st.session_state.boundary_points,
            color='red',
            weight=2,
            fill=False,
            popup="Selected Area"
        ).add_to(m)
    
    # Display protected areas
    for i, area in enumerate(st.session_state.protected_areas):
        folium.Polygon(
            locations=area['points'],
            color='green',
            weight=2,
            fill=True,
            fill_color='green',
            fill_opacity=0.3,
            tooltip=area.get('name', f"Protected Area {i+1}")
        ).add_to(m)
    
    # Display sensor locations
    for i, location in enumerate(st.session_state.potential_locations):
        sensor_name = location.get('name', f"Sensor {i+1}")
        folium.Marker(
            location=[location['lat'], location['lng']],
            popup=sensor_name,
            tooltip=sensor_name,
            icon=folium.Icon(color='blue', icon='circle', prefix='fa')
        ).add_to(m)
    
    # Add colored grid based on Pd values - USE EXACT GRID LOGIC
    add_colored_grid_to_map(m, model_results, target_type, selection_level, selected_option, classification_type)
    
    return m

def is_point_in_rectangle(point, sw_corner, ne_corner):
    """Check if a point is inside a rectangle"""
    lat, lng = point
    return (sw_corner[0] <= lat <= ne_corner[0]) and (sw_corner[1] <= lng <= ne_corner[1])

def is_point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm"""
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

def add_colored_grid_to_map(map_obj, model_results, target_type, selection_level, selected_option, classification_type):
    """
    Add colored grid cells to the map based on Pd values
    Shows ALL grid cells (like map_selection.py) but only COLORS the ones inside the boundary
    """
    try:
        # Get grid configuration from session state
        if not st.session_state.grid_enabled:
            st.warning("Grid is not enabled. Please enable grid in Map & Selection tab.")
            return
            
        sw_corner = st.session_state.sw_corner
        ne_corner = st.session_state.ne_corner
        grid_size_degrees = st.session_state.grid_size_degrees
        
        # Extract Pd values from model results
        pd_values = extract_pd_values(model_results, target_type, selection_level, selected_option, classification_type)
        
        if not pd_values:
            st.warning("No Pd values found in model results.")
            return
        
        # Create ALL grid cells (like map_selection.py) with inside/outside info
        all_grid_cells, num_rows, num_cols, inside_count = create_all_grid_cells_with_inside_info(
            sw_corner, ne_corner, grid_size_degrees
        )
        
        # Determine color based on classification type
        if classification_type == "target_detection":
            fill_color = "green"
        elif classification_type == "foe_or_friend":
            fill_color = "yellow"
        else:
            fill_color = "green"  # Default
        
        # Draw ALL grid cells
        for cell in all_grid_cells:
            if cell['is_inside']:
                # Cell is INSIDE boundary - can be colored based on Pd value
                pd_array_index = cell['pd_array_index']
                
                if pd_array_index is not None and pd_array_index < len(pd_values):
                    pd_value = pd_values[pd_array_index]
                    
                    if pd_value > 0:
                        # Color based on Pd value
                        if pd_value <= 0.3:
                            opacity = 0.2
                        elif pd_value <= 0.5:
                            opacity = 0.4
                        elif pd_value <= 0.7:
                            opacity = 0.6
                        elif pd_value <= 0.8:
                            opacity = 0.75
                        else:
                            opacity = 0.9
                        
                        folium.Rectangle(
                            bounds=cell['bounds'],
                            color='black',
                            weight=1,
                            fill=True,
                            fill_color=fill_color,
                            fill_opacity=opacity,
                            popup=f"INSIDE - Grid Cell ({cell['row']},{cell['col']})<br>ID: r{cell['row']}c{cell['col']}<br>Pd: {pd_value:.3f}<br>Type: {classification_type}<br>Array Index: {pd_array_index}"
                        ).add_to(map_obj)
                    else:
                        # Inside but no detection (Pd = 0)
                        folium.Rectangle(
                            bounds=cell['bounds'],
                            color='black',
                            weight=1,
                            fill=False,
                            opacity=0.7,
                            popup=f"INSIDE - Grid Cell ({cell['row']},{cell['col']})<br>ID: r{cell['row']}c{cell['col']}<br>Pd: {pd_value:.3f}<br>No detection<br>Array Index: {pd_array_index}"
                        ).add_to(map_obj)
                else:
                    # Inside but no data (shouldn't happen)
                    folium.Rectangle(
                        bounds=cell['bounds'],
                        color='red',
                        weight=1,
                        fill=False,
                        opacity=0.7,
                        popup=f"INSIDE - ERROR: No data for cell ({cell['row']},{cell['col']})"
                    ).add_to(map_obj)
            else:
                # Cell is OUTSIDE boundary - show as empty grid (no color, just border)
                folium.Rectangle(
                    bounds=cell['bounds'],
                    color='black',
                    weight=1,
                    fill=False,
                    opacity=0.3,  # Lighter opacity for outside cells
                    popup=f"OUTSIDE - Grid Cell ({cell['row']},{cell['col']})<br>ID: r{cell['row']}c{cell['col']}<br>Center outside boundary<br>No Pd data"
                ).add_to(map_obj)
        
        # Show statistics
        # st.info(f"Displaying {len(all_grid_cells)} total grid cells ({inside_count} inside boundary, {len(all_grid_cells)-inside_count} outside). Pd array has {len(pd_values)} values.")
        
    except Exception as e:
        st.error(f"Error adding colored grid: {str(e)}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")

def extract_pd_values(model_results, target_type, selection_level, selected_option, classification_type):
    """
    Extract Pd values from model results based on target type, selection level and option
    
    Parameters:
    -----------
    model_results : dict
        The output from the analytical model
    target_type : str
        The selected target type (UAV configuration)
    selection_level : str
        "all_sensors", "by_type", or "by_individual"
    selected_option : str
        The specific option selected (sensor type or sensor name)
    classification_type : str
        Either "target_detection" or "foe_or_friend"
    """
    try:
        # Navigate through the model results structure
        per_square_metrics = model_results.get("per_square_metrics", {})
        
        if target_type not in per_square_metrics:
            return []
        
        target_data = per_square_metrics[target_type]
        
        # Extract data based on selection level
        if selection_level == "all_sensors":
            sensor_data = target_data.get("for_all_sensors", {})
        elif selection_level == "by_type":
            sensor_data = target_data.get("by_sensor_type", {}).get(selected_option, {})
        elif selection_level == "by_individual":
            sensor_data = target_data.get("by_sensor", {}).get(selected_option, {})
        else:
            return []
        
        classification_data = sensor_data.get(classification_type, {})
        pd_values = classification_data.get("Pd", [])
        
        return pd_values if isinstance(pd_values, list) else []
        
    except Exception as e:
        st.error(f"Error extracting Pd values: {str(e)}")
        return []

def add_color_legend_to_map(map_obj, classification_type):
    """
    Add a color legend to the map showing Pd value scale
    """
    if classification_type == "target_detection":
        legend_color = "yellow"
        legend_title = "Target Detection (Pd)"
    elif classification_type == "foe_or_friend":
        legend_color = "green"
        legend_title = "Friend or Foe (Pd)"
    else:
        legend_color = "yellow"
        legend_title = "Detection Probability (Pd)"
    
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 180px; height: 110px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>{legend_title}</b></p>
    <p><i class="fa fa-square" style="color:{legend_color}; opacity:0.8"></i> High (0.7-1.0)</p>
    <p><i class="fa fa-square" style="color:{legend_color}; opacity:0.5"></i> Medium (0.4-0.7)</p>
    <p><i class="fa fa-square" style="color:{legend_color}; opacity:0.2"></i> Low (0.1-0.4)</p>
    <p><i class="fa fa-square" style="color:black; opacity:0.3"></i> No Detection (0.0)</p>
    </div>
    '''
    map_obj.get_root().html.add_child(folium.Element(legend_html))

def get_available_sensors_from_results(model_results, target_type):
    """
    Extract available sensor options from model results for a specific target type
    Returns organized structure with all three levels: all_sensors, by_type, by_individual
    """
    try:
        per_square_metrics = model_results.get("per_square_metrics", {})
        
        if target_type not in per_square_metrics:
            return {"all_sensors": [], "by_type": {}, "by_individual": []}
        
        target_data = per_square_metrics[target_type]
        result = {
            "all_sensors": [],
            "by_type": {},
            "by_individual": []
        }
        
        # 1. For All Sensors option
        if "for_all_sensors" in target_data:
            result["all_sensors"] = ["for_all_sensors"]
        
        # 2. By Sensor Type options
        by_sensor_type = target_data.get("by_sensor_type", {})
        for sensor_type in by_sensor_type.keys():
            result["by_type"][sensor_type] = sensor_type
        
        # 3. By Individual Sensor options
        by_sensor = target_data.get("by_sensor", {})
        for sensor_id in by_sensor.keys():
            result["by_individual"].append(sensor_id)
        
        return result
        
    except Exception as e:
        st.error(f"Error extracting sensor options: {str(e)}")
        return {"all_sensors": [], "by_type": {}, "by_individual": []}

def format_target_type_for_display(target_type):
    """
    Format target type string for better display
    Examples: 
    - "fixed_wing_alt_2_7_speed_151" → "Fixed Wing (Alt: 2.7km, Speed: 151km/h)"
    - "quadcopter_alt_0_5_speed_72" → "Quadcopter (Alt: 0.5km, Speed: 72km/h)"
    """
    if not target_type:
        return "Unknown UAV"
    
    # Parse the target type string
    parts = target_type.split('_')
    
    try:
        # Find UAV type (everything before 'alt')
        uav_type_parts = []
        for i, part in enumerate(parts):
            if part == 'alt':
                break
            uav_type_parts.append(part)
        
        uav_type = ' '.join(uav_type_parts).title()
        
        # Extract altitude and speed
        altitude = None
        speed = None
        
        for i, part in enumerate(parts):
            if part == 'alt' and i + 1 < len(parts):
                # Next part should be altitude (might have underscores for decimals)
                alt_part = parts[i + 1]
                if i + 2 < len(parts) and parts[i + 2].isdigit():
                    # Handle decimal like "2_7" → "2.7"
                    alt_part = alt_part + '.' + parts[i + 2]
                altitude = alt_part
                
            elif part == 'speed' and i + 1 < len(parts):
                speed = parts[i + 1]
        
        # Format the display string
        if altitude and speed:
            return f"{uav_type} (Alt: {altitude}km, Speed: {speed}km/h)"
        elif altitude:
            return f"{uav_type} (Alt: {altitude}km)"
        elif speed:
            return f"{uav_type} (Speed: {speed}km/h)"
        else:
            return uav_type
            
    except Exception:
        # Fallback to simple formatting
        return target_type.replace("_", " ").title()

def get_available_target_types(model_results):
    """
    Extract available target types from model results
    """
    try:
        per_square_metrics = model_results.get("per_square_metrics", {})
        target_types = list(per_square_metrics.keys())
        return sorted(target_types) if target_types else []
    except Exception as e:
        st.error(f"Error extracting target types: {str(e)}")
        return []

def format_sensor_name_for_display(sensor_id):
    """
    Format sensor ID for better display
    """
    if sensor_id == "for_all_sensors":
        return "For All Sensors"
    else:
        # Clean up sensor names - replace underscores with spaces, capitalize
        formatted = sensor_id.replace("_", " ").title()
        return formatted

def prediction_results_tab():
    """
    Main function for the prediction results visualization tab
    """
    st.header("Prediction Results Visualization")
    
    # Check if we have model results
    if not st.session_state.get('model_completed', False) or not st.session_state.get('model_results'):
        st.warning("No model results available. Please run the prediction model first.")
        return
    
    model_results = st.session_state.model_results
    
    # Create layout: Map on left, controls on right
    map_col, control_col = st.columns([3, 1])
    
    with control_col:
        st.subheader("Display Options")
        
        # Target type selector - FIRST get available targets
        available_targets = get_available_target_types(model_results)
        if not available_targets:
            st.error("No target types available for visualization.")
            return
        
        target_type = st.selectbox(
            "Select Target Type (UAV Configuration):",
            available_targets,
            format_func=lambda x: format_target_type_for_display(x),
            help="Different UAV configurations with varying altitude and speed parameters"
        )
        
        # NOW get available sensors for the selected target type
        available_options = get_available_sensors_from_results(model_results, target_type)
        
        # Step 1: Selection Level
        selection_level = st.selectbox(
            "Select Analysis Level:",
            ["all_sensors", "by_type", "by_individual"],
            format_func=lambda x: {
                "all_sensors": "For All Sensors",
                "by_type": "By Sensor Type", 
                "by_individual": "By Individual Sensor"
            }[x],
            index=0  # Default to "For All Sensors"
        )
        
        # Step 2: Specific Option Selection based on level
        if selection_level == "all_sensors":
            selected_option = "for_all_sensors"
            st.info("Showing combined results from all sensors")
            
        elif selection_level == "by_type":
            if available_options["by_type"]:
                sensor_types = list(available_options["by_type"].keys())
                selected_option = st.selectbox(
                    "Select Sensor Type:",
                    sensor_types,
                    format_func=lambda x: x.capitalize()
                )
            else:
                st.warning("No sensor type data available")
                selected_option = None
                
        elif selection_level == "by_individual":
            if available_options["by_individual"]:
                selected_option = st.selectbox(
                    "Select Individual Sensor:",
                    available_options["by_individual"],
                    format_func=format_sensor_name_for_display
                )
            else:
                st.warning("No individual sensor data available")
                selected_option = None
        
        # Classification type selector
        classification_type = st.selectbox(
            "Select Classification:",
            ["target_detection", "foe_or_friend"],
            format_func=lambda x: "Target Detection" if x == "target_detection" else "Friend or Foe"
        )
        
        # Display current selection info
        st.markdown("---")
        st.markdown("**Current Selection:**")
        if selection_level == "all_sensors":
            st.write(f"**Level:** For All Sensors")
        elif selection_level == "by_type":
            st.write(f"**Level:** By Sensor Type")
            st.write(f"**Type:** {selected_option.capitalize() if selected_option else 'None'}")
        elif selection_level == "by_individual":
            st.write(f"**Level:** Individual Sensor")
            st.write(f"**Sensor:** {format_sensor_name_for_display(selected_option) if selected_option else 'None'}")
        
        st.write(f"**Classification:** {'Target Detection' if classification_type == 'target_detection' else 'Friend or Foe'}")
        st.write(f"**Target:** {format_target_type_for_display(target_type)}")
        
        # Show some statistics if we have a valid selection
        if selected_option:
            pd_values = extract_pd_values(model_results, target_type, selection_level, selected_option, classification_type)
            if pd_values:
                non_zero_values = [pd for pd in pd_values if pd > 0]
                if non_zero_values:
                    st.markdown("**Statistics:**")
                    st.write(f"Max Pd: {max(non_zero_values):.3f}")
                    st.write(f"Avg Pd: {np.mean(non_zero_values):.3f}")
                    st.write(f"Coverage: {len(non_zero_values)}/{len(pd_values)} cells")
        
        # Download results button
        st.markdown("---")
        if st.button("Download Prediction Output", type="primary"):
            results_json = json.dumps(model_results, indent=2)
            instance_id = st.session_state.get('custom_instance_id', 'default')
            
            st.download_button(
                label="Download Prediction Output",
                data=results_json,
                file_name=f"out_prediction_{instance_id}.json",
                mime="application/json",
                key="download_prediction_output_final"
            )
    
    with map_col:
        st.subheader("Detection Probability Map")
        
        # Check if grid is enabled
        if not st.session_state.get('grid_enabled', False):
            st.warning("Grid visualization requires grid to be enabled in Map & Selection tab.")
            return
        
        # Create and display the map
        try:
            if selected_option:
                prediction_map = create_prediction_results_map(
                    model_results, 
                    target_type,
                    selection_level,
                    selected_option, 
                    classification_type
                )
                folium_static(prediction_map, width=800, height=600)
            else:
                st.warning("Please select a valid option to display the map.")
            
        except Exception as e:
            st.error(f"Error creating map: {str(e)}")

# Optional: Function to integrate into existing prediction.py
def integrate_with_prediction_tab():
    """
    This function can be called from prediction.py to show results
    """
    if st.session_state.get('model_completed', False) and st.session_state.get('model_results'):
        st.markdown("---")
        prediction_results_tab()