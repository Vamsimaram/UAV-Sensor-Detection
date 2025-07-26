import streamlit as st
import folium
from streamlit_folium import folium_static, st_folium
import pandas as pd
import json
import numpy as np
from shapely.geometry import Point, Polygon

# Import from map_utils
from map_utils import create_boundary_drawing_map

def initialize_session_state():
    """Initialize all required session state variables"""
    if 'area_selected' not in st.session_state:
        st.session_state.area_selected = False
    if 'map_center' not in st.session_state:
        st.session_state.map_center = [37.7749, -122.4194]  # Default San Francisco
    if 'grid_enabled' not in st.session_state:
        st.session_state.grid_enabled = False
    if 'grid_size_degrees' not in st.session_state:
        st.session_state.grid_size_degrees = 0.003
    if 'boundary_type' not in st.session_state:
        st.session_state.boundary_type = "rectangle"
    if 'boundary_points' not in st.session_state:
        st.session_state.boundary_points = []
    if 'sw_corner' not in st.session_state:
        st.session_state.sw_corner = None
    if 'ne_corner' not in st.session_state:
        st.session_state.ne_corner = None
    if 'potential_locations' not in st.session_state:
        st.session_state.potential_locations = []
    if 'protected_areas' not in st.session_state:
        st.session_state.protected_areas = []
    if 'coord_input_list' not in st.session_state:
        st.session_state.coord_input_list = []
    if 'last_drawn_feature' not in st.session_state:
        st.session_state.last_drawn_feature = None
    if 'map_data' not in st.session_state:
        st.session_state.map_data = None
    # Add flags for better control
    if 'drawing_ready' not in st.session_state:
        st.session_state.drawing_ready = False
    if 'force_map_update' not in st.session_state:
        st.session_state.force_map_update = False

@st.cache_data
def create_cached_grid(sw_corner_tuple, ne_corner_tuple, grid_size_degrees):
    """
    Cached function to create grid coordinates - prevents recalculation
    """
    sw_corner = list(sw_corner_tuple)
    ne_corner = list(ne_corner_tuple)
    
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
    
    grid_cells = []
    for i in range(num_rows):
        for j in range(num_cols):
            # Calculate the corners of this cell
            lat_sw = sw_corner[0] + i * lat_grid_size
            lng_sw = sw_corner[1] + j * lng_grid_size
            
            lat_ne = lat_sw + lat_grid_size
            lng_ne = lng_sw + lng_grid_size
            
            grid_cells.append({
                'bounds': [[lat_sw, lng_sw], [lat_ne, lng_ne]],
                'row': i,
                'col': j,
                'lat_size': lat_grid_size,
                'lng_size': lng_grid_size
            })
    
    return grid_cells, num_rows, num_cols

def add_grid_to_map(map_obj, grid_cells, grid_size_degrees):
    """Add grid overlay to map using pre-calculated cells"""
    approx_km = grid_size_degrees * 111.0  # 1 degree ≈ 111 km
    
    for cell in grid_cells:
        folium.Rectangle(
            bounds=cell['bounds'],
            color='black',
            weight=1,
            fill=False,
            opacity=0.7,
            popup=f"Grid Cell ({cell['row']},{cell['col']})<br>ID: r{cell['row']}c{cell['col']}<br>Lat: {cell['lat_size']:.6f}°<br>Lng: {cell['lng_size']:.6f}°<br>~{approx_km:.2f}km × {approx_km:.2f}km"
        ).add_to(map_obj)

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

def degrees_to_km_approximate(degrees, latitude=None):
    """Convert degrees to approximate kilometers"""
    lat_km = degrees * 111.0
    
    if latitude is not None:
        lng_km = degrees * 111.0 * np.cos(np.radians(latitude))
    else:
        lng_km = degrees * 111.0 * 0.707
    
    return lat_km, lng_km

def calculate_area_dimensions_degrees(sw_corner, ne_corner):
    """Calculate the width and height of an area in degrees"""
    width_degrees = ne_corner[1] - sw_corner[1]
    height_degrees = ne_corner[0] - sw_corner[0]
    return width_degrees, height_degrees

def render_grid_statistics(sw_corner, ne_corner, grid_size_degrees):
    """Render grid statistics in a more efficient way"""
    # Get center latitude for calculations
    center_lat = (sw_corner[0] + ne_corner[0]) / 2
    
    # Calculate adjusted longitude grid size for square cells
    lat_grid_size = grid_size_degrees
    lng_grid_size = grid_size_degrees / np.cos(np.radians(center_lat))
    
    # Calculate dimensions in degrees
    width_degrees, height_degrees = calculate_area_dimensions_degrees(sw_corner, ne_corner)
    
    # Calculate number of cells in each dimension
    num_rows = int(np.ceil(height_degrees / lat_grid_size))
    num_cols = int(np.ceil(width_degrees / lng_grid_size))
    total_cells = num_rows * num_cols
    
    # Calculate area coverage in degrees
    selected_area_deg2 = width_degrees * height_degrees
    grid_coverage_deg2 = num_rows * num_cols * (lat_grid_size * lng_grid_size)
    
    # Convert to approximate kilometers for reference
    lat_km_per_deg, lng_km_per_deg = degrees_to_km_approximate(1.0, center_lat)
    
    # Calculate approximate dimensions in km (should be equal for square cells)
    grid_cell_side_km = grid_size_degrees * lat_km_per_deg  # Physical size of each side
    grid_cell_size_km2_approx = grid_cell_side_km ** 2  # Square area
    
    width_km_approx = width_degrees * lng_km_per_deg
    height_km_approx = height_degrees * lat_km_per_deg
    selected_area_km2_approx = width_km_approx * height_km_approx
    
    # Display statistics in an expander
    with st.expander("Grid Statistics", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Grid Cell Size (degrees)", f"{grid_size_degrees:.6f}° lat", 
                    help="Latitude size - longitude is adjusted for square cells")
            st.metric("Grid Cell Size (approx km)", f"{grid_cell_size_km2_approx:.3f} km²", 
                    help="Approximate square area in kilometers")
            st.metric("Grid Dimensions", f"{num_rows} × {num_cols}", 
                    help="Rows × Columns")
            st.metric("Total Grid Cells", f"{total_cells:,}")
        
        with col2:
            st.metric("Selected Area (degrees)", f"{selected_area_deg2:.6f}°²")
            st.metric("Selected Area (approx km)", f"{selected_area_km2_approx:.1f} km²", 
                    help="Approximate area in kilometers")
            st.metric("Area Dimensions (degrees)", f"{width_degrees:.6f}° × {height_degrees:.6f}°", 
                    help="Width × Height in degrees")
            coverage_ratio = (selected_area_deg2 / grid_coverage_deg2) * 100 if grid_coverage_deg2 > 0 else 0
            st.metric("Grid Coverage Efficiency", f"{coverage_ratio:.1f}%",
                    help="Percentage of grid area that overlaps with selected area")

def process_drawing_data(map_data):
    """Process drawing data more efficiently"""
    if not map_data or 'all_drawings' not in map_data or not map_data['all_drawings']:
        return False, None, None, None, None
    
    try:
        last_drawing = map_data['all_drawings'][-1]
        
        if 'geometry' not in last_drawing or 'coordinates' not in last_drawing['geometry']:
            return False, None, None, None, None
        
        coords = last_drawing['geometry']['coordinates']
        
        # Process rectangle shapes
        if (last_drawing['geometry']['type'] == 'Rectangle' or 
            (last_drawing['geometry']['type'] == 'Polygon' and coords and len(coords[0]) == 5)):
            
            if coords and coords[0]:
                boundary_points = []
                for coord in coords[0]:
                    # Convert [lng, lat] to [lat, lng]
                    boundary_points.append([coord[1], coord[0]])
                
                # Calculate bounding box
                lats = [p[0] for p in boundary_points]
                lngs = [p[1] for p in boundary_points]
                
                sw_corner = [min(lats), min(lngs)]
                ne_corner = [max(lats), max(lngs)]
                map_center = [(min(lats) + max(lats))/2, (min(lngs) + max(lngs))/2]
                
                return True, sw_corner, ne_corner, boundary_points, map_center
    except Exception as e:
        st.error(f"Error processing drawing: {e}")
        return False, None, None, None, None
    
    return False, None, None, None, None

def map_selection_tab():
    """Map-based selection tab for defining area of interest"""
    
    # Initialize session state first
    initialize_session_state()
    
    # Default starting coordinate (San Francisco)
    default_coordinate = [37.7749, -122.4194]

    st.header("Select Area of Interest")
    
    # Grid configuration section - Only show if an area is selected
    if st.session_state.area_selected:
        st.markdown("---")
        st.subheader("Grid Configuration")
        
        # Add grid configuration controls
        enable_grid = st.checkbox("Enable Grid Overlay", value=st.session_state.grid_enabled)
        
        if enable_grid:
            # Grid size control - input in degrees
            grid_size_degrees = st.number_input(
                "Grid Square Side Length (degrees)",
                min_value=0.0001,
                max_value=1.0, 
                value=st.session_state.grid_size_degrees,
                step=0.0001,
                format="%.6f",
                help="Size of each grid square in degrees. Grid cells will be squares in degree units."
            )
            
            # Store grid settings in session state
            st.session_state.grid_enabled = enable_grid
            st.session_state.grid_size_degrees = grid_size_degrees
            
            # Calculate grid statistics only if we have valid corners
            if (st.session_state.area_selected and 
                st.session_state.sw_corner is not None and 
                st.session_state.ne_corner is not None):
                
                render_grid_statistics(
                    st.session_state.sw_corner, 
                    st.session_state.ne_corner, 
                    grid_size_degrees
                )
        else:
            st.session_state.grid_enabled = False
    
    st.markdown("---")
    
    # Selection method radio buttons
    selection_method = st.radio(
        "Select area by:",
        ["Drawing on map", "Entering coordinates"]
    )
    
    if selection_method == "Entering coordinates":
        st.subheader("Enter Coordinates")
        st.info("Enter coordinates in latitude, longitude format")
        
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
            
            add_point = st.form_submit_button("Add Point")
            
            if add_point:
                st.session_state.coord_input_list.append([new_lat, new_lng])
                st.rerun()
        
        # Remove and clear buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Remove Last Point") and st.session_state.coord_input_list:
                st.session_state.coord_input_list.pop()
                st.rerun()
        
        with col2:
            if st.button("Clear All Points"):
                st.session_state.coord_input_list = []
                st.rerun()
        
        # Create preview map
        st.subheader("Preview Map")
        
        # Determine map center
        if st.session_state.coord_input_list:
            lats = [coord[0] for coord in st.session_state.coord_input_list]
            lngs = [coord[1] for coord in st.session_state.coord_input_list]
            map_center = [(min(lats) + max(lats))/2, (min(lngs) + max(lngs))/2]
            zoom_start = 10
        else:
            map_center = st.session_state.map_center
            zoom_start = 12
        
        # Create preview map
        preview_map = folium.Map(location=map_center, zoom_start=zoom_start)
        
        # Add coordinate points and lines
        if st.session_state.coord_input_list:
            # Add markers for each point
            for i, coord in enumerate(st.session_state.coord_input_list):
                folium.Marker(
                    location=coord,
                    popup=f"Point {i+1}: {coord[0]:.6f}, {coord[1]:.6f}",
                    icon=folium.Icon(icon="map-pin")
                ).add_to(preview_map)
            
            # Connect points with lines
            if len(st.session_state.coord_input_list) >= 2:
                folium.PolyLine(
                    locations=st.session_state.coord_input_list,
                    color='blue',
                    weight=3,
                    opacity=0.7
                ).add_to(preview_map)
            
            # Draw polygon if 3+ points
            if len(st.session_state.coord_input_list) >= 3:
                draw_polygon_on_map(
                    preview_map, 
                    st.session_state.coord_input_list, 
                    color='blue',
                    fill_opacity=0.2,
                    tooltip="Current Points Polygon"
                )
        
        # Show existing selected area
        if (st.session_state.area_selected and 
            st.session_state.sw_corner is not None and 
            st.session_state.ne_corner is not None):
            
            if st.session_state.boundary_type == "rectangle":
                folium.Rectangle(
                    bounds=[[st.session_state.sw_corner[0], st.session_state.sw_corner[1]], 
                          [st.session_state.ne_corner[0], st.session_state.ne_corner[1]]],
                    color='red',
                    fill=True,
                    fill_opacity=0.2,
                    tooltip="Current Selected Area"
                ).add_to(preview_map)
            elif st.session_state.boundary_type == "polygon" and st.session_state.boundary_points:
                draw_polygon_on_map(
                    preview_map, 
                    st.session_state.boundary_points, 
                    color='red',
                    fill_opacity=0.2,
                    tooltip="Current Selected Area"
                )
            
            # Add grid if enabled (using cached function)
            if st.session_state.grid_enabled:
                try:
                    grid_cells, num_rows, num_cols = create_cached_grid(
                        tuple(st.session_state.sw_corner),
                        tuple(st.session_state.ne_corner),
                        st.session_state.grid_size_degrees
                    )
                    add_grid_to_map(preview_map, grid_cells, st.session_state.grid_size_degrees)
                except Exception as e:
                    st.warning(f"Could not display grid: {e}")
        
        # Display the preview map
        folium_static(preview_map, width=1000, height=500)
        
        # Set Area button
        if len(st.session_state.coord_input_list) >= 3:
            if st.button("Set Area from Coordinates"):
                coordinates = st.session_state.coord_input_list
                
                # Calculate bounding box
                lats = [coord[0] for coord in coordinates]
                lngs = [coord[1] for coord in coordinates]
                
                sw_corner = [min(lats), min(lngs)]
                ne_corner = [max(lats), max(lngs)]
                
                # Update session state
                st.session_state.sw_corner = sw_corner
                st.session_state.ne_corner = ne_corner
                st.session_state.boundary_type = "polygon"
                st.session_state.boundary_points = coordinates
                st.session_state.area_selected = True
                st.session_state.map_center = [(min(lats) + max(lats))/2, 
                                            (min(lngs) + max(lngs))/2]
                
                st.success("Area set successfully!")
                st.rerun()
        else:
            st.warning("Add at least 3 points to define an area.")
    
    # Drawing on map method - OPTIMIZED VERSION
    elif selection_method == "Drawing on map":
        
        # Show status immediately
        if st.session_state.area_selected:
            st.info("Your current selected area is shown in red. Draw a new rectangle to update it.")
        else:
            st.info("Use the rectangle tool in the top right corner of the map to select an area.")
        
        # Show button first, before map rendering
        st.markdown("---")
        
        # Create columns for buttons
        col1, col2 = st.columns([1, 1])
        
        with col1:
            set_area_clicked = st.button("Set Drawn Area", key="set_drawn_area_btn", type="primary")
        
        with col2:
            clear_button_clicked = st.button("Clear Area", key="clear_area_btn", type="secondary")
        
        # Handle button clicks immediately
        if clear_button_clicked:
            # Reset all relevant session state
            st.session_state.area_selected = False
            st.session_state.boundary_type = "rectangle"
            st.session_state.boundary_points = []
            st.session_state.sw_corner = None
            st.session_state.ne_corner = None
            st.session_state.potential_locations = []
            st.session_state.protected_areas = []
            st.session_state.coord_input_list = []
            st.session_state.grid_enabled = False
            st.session_state.last_drawn_feature = None
            st.session_state.map_data = None
            st.session_state.drawing_ready = False
            
            st.success("Area cleared. All sensors, protected areas, and grid settings were also cleared.")
            st.rerun()
        
        # Create drawing map
        try:
            drawing_map = create_boundary_drawing_map(
                st.session_state.map_center,
                zoom_start=12,
                predefined_locations={}
            )
            
            # Show existing selected area
            if (st.session_state.area_selected and 
                st.session_state.sw_corner is not None and 
                st.session_state.ne_corner is not None):
                
                if st.session_state.boundary_type == "rectangle":
                    folium.Rectangle(
                        bounds=[[st.session_state.sw_corner[0], st.session_state.sw_corner[1]], 
                              [st.session_state.ne_corner[0], st.session_state.ne_corner[1]]],
                        color='red',
                        fill=True,
                        fill_opacity=0.2,
                        tooltip="Current Selected Area"
                    ).add_to(drawing_map)
                elif st.session_state.boundary_type == "polygon" and st.session_state.boundary_points:
                    draw_polygon_on_map(
                        drawing_map, 
                        st.session_state.boundary_points, 
                        color='red',
                        fill_opacity=0.2,
                        tooltip="Current Selected Area"
                    )
                
                # Add grid if enabled (using cached function)
                if st.session_state.grid_enabled:
                    try:
                        grid_cells, num_rows, num_cols = create_cached_grid(
                            tuple(st.session_state.sw_corner),
                            tuple(st.session_state.ne_corner),
                            st.session_state.grid_size_degrees
                        )
                        add_grid_to_map(drawing_map, grid_cells, st.session_state.grid_size_degrees)
                    except Exception as e:
                        st.warning(f"Could not display grid: {e}")
            
            # Display map with drawing controls - reduced height for faster rendering
            map_data = st_folium(drawing_map, width=1000, height=500, key="drawing_map")
            
            # Process drawn features more efficiently
            if map_data:
                drawing_detected, sw_corner, ne_corner, boundary_points, map_center = process_drawing_data(map_data)
                
                if drawing_detected:
                    # Update session state immediately
                    st.session_state.sw_corner = sw_corner
                    st.session_state.ne_corner = ne_corner
                    st.session_state.boundary_type = "rectangle"
                    st.session_state.boundary_points = boundary_points
                    st.session_state.map_center = map_center
                    st.session_state.drawing_ready = True
                    
                    # Show preview of coordinates
                    st.success(f"Rectangle drawn! SW: {sw_corner}, NE: {ne_corner}")
            
            # Handle set area button click
            if set_area_clicked:
                if (hasattr(st.session_state, 'sw_corner') and 
                    st.session_state.sw_corner is not None and 
                    st.session_state.ne_corner is not None):
                    
                    # Clear existing data when setting new area
                    st.session_state.potential_locations = []
                    st.session_state.protected_areas = []
                    st.session_state.map_data = None
                    
                    st.session_state.area_selected = True
                    st.success("Area set successfully!")
                    st.rerun()
                else:
                    st.error("Please draw a rectangle on the map first.")
                    
        except Exception as e:
            st.error(f"Error creating map: {e}")
            st.info("Please refresh the page if the map doesn't load properly.")

# Example usage - call this function in your Streamlit app
if __name__ == "__main__":
    map_selection_tab()