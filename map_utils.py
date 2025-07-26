import folium
from folium.plugins import Draw
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import math

def calculate_distance_km(lat1, lng1, lat2, lng2):
    """
    Calculate distance between two points using Haversine formula
    Returns distance in kilometers
    
    Parameters:
    -----------
    lat1, lng1 : float
        Latitude and longitude of first point
    lat2, lng2 : float
        Latitude and longitude of second point
        
    Returns:
    --------
    float : Distance in kilometers
    """
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lng1_rad = math.radians(lng1)
    lat2_rad = math.radians(lat2)
    lng2_rad = math.radians(lng2)
    
    # Haversine formula
    earth_radius = 6371  # Earth radius in kilometers
    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = earth_radius * c  # Distance in kilometers
    
    return distance

def calculate_grid_dimensions_km(sw_corner, ne_corner, grid_size_degrees):
    """
    Calculate the actual dimensions of grid cells in kilometers
    
    Parameters:
    -----------
    sw_corner : list
        [lat, lng] of southwest corner
    ne_corner : list
        [lat, lng] of northeast corner
    grid_size_degrees : float
        Grid size in degrees
        
    Returns:
    --------
    dict : Dictionary with grid dimensions and statistics
    """
    # Calculate center point for more accurate measurements
    center_lat = (sw_corner[0] + ne_corner[0]) / 2
    center_lng = (sw_corner[1] + ne_corner[1]) / 2
    
    # Calculate horizontal distance (1 degree longitude at center latitude)
    lng_distance_km = calculate_distance_km(
        center_lat, center_lng,
        center_lat, center_lng + grid_size_degrees
    )
    
    # Calculate vertical distance (1 degree latitude)
    lat_distance_km = calculate_distance_km(
        center_lat, center_lng,
        center_lat + grid_size_degrees, center_lng
    )
    
    # Calculate total area coverage
    total_lat_range = ne_corner[0] - sw_corner[0]
    total_lng_range = ne_corner[1] - sw_corner[1]
    
    num_rows = int(total_lat_range / grid_size_degrees)
    num_cols = int(total_lng_range / grid_size_degrees)
    
    # Calculate total area in square kilometers
    total_width_km = calculate_distance_km(
        center_lat, sw_corner[1],
        center_lat, ne_corner[1]
    )
    total_height_km = calculate_distance_km(
        sw_corner[0], center_lng,
        ne_corner[0], center_lng
    )
    
    return {
        'grid_cell_width_km': lng_distance_km,
        'grid_cell_height_km': lat_distance_km,
        'grid_cell_area_km2': lng_distance_km * lat_distance_km,
        'total_grid_cells': num_rows * num_cols,
        'total_area_km2': total_width_km * total_height_km,
        'grid_rows': num_rows,
        'grid_cols': num_cols
    }

def create_square_grid_overlay(map_obj, sw_corner, ne_corner, grid_size):
    """
    Create a grid overlay on the map with square-shaped cells
    
    Parameters:
    -----------
    map_obj : folium.Map
        The map object to add the grid to
    sw_corner : list
        [lat, lng] of southwest corner
    ne_corner : list
        [lat, lng] of northeast corner
    grid_size : float
        Size of grid squares in degrees (same for both lat and lng to ensure squares)
    """
    # Calculate grid dimensions for display
    grid_info = calculate_grid_dimensions_km(sw_corner, ne_corner, grid_size)
    
    # Calculate number of rows and columns in the grid
    lat_range = ne_corner[0] - sw_corner[0]
    lng_range = ne_corner[1] - sw_corner[1]
    
    num_rows = int(lat_range / grid_size)
    num_cols = int(lng_range / grid_size)
    
    # Create grid cell for each row and column
    for i in range(num_rows):
        for j in range(num_cols):
            lat_sw = sw_corner[0] + i * grid_size
            lng_sw = sw_corner[1] + j * grid_size
            
            lat_ne = lat_sw + grid_size
            lng_ne = lng_sw + grid_size
            
            # Add rectangle for each grid cell with better styling
            folium.Rectangle(
                bounds=[[lat_sw, lng_sw], [lat_ne, lng_ne]],
                color='black',       # Black outline for better visibility
                weight=1,            # Thinner lines to avoid cluttering
                fill=False,          # No fill to see the map underneath
                opacity=0.7,         # Slightly transparent
                popup=f"Grid Cell ({i},{j})<br>ID: r{i}c{j}<br>SW: [{lat_sw:.6f}, {lng_sw:.6f}]<br>NE: [{lat_ne:.6f}, {lng_ne:.6f}]<br>Dimensions: {grid_info['grid_cell_width_km']:.2f} x {grid_info['grid_cell_height_km']:.2f} km<br>Area: {grid_info['grid_cell_area_km2']:.2f} km²"
            ).add_to(map_obj)

def create_grid_overlay(map_obj, sw_corner, ne_corner, grid_size):
    """
    Create a grid overlay on the map based on specified corners and grid size
    This function is kept for backward compatibility, but create_square_grid_overlay is preferred
    for square grids
    
    Parameters:
    -----------
    map_obj : folium.Map
        The map object to add the grid to
    sw_corner : list
        [lat, lng] of southwest corner
    ne_corner : list
        [lat, lng] of northeast corner
    grid_size : float or list
        Size of grid squares in degrees
        Can be a single float (same size for lat/lng) or 
        a list [lat_size, lng_size] for different sizes
    """
    # Handle different grid size formats
    if isinstance(grid_size, list) and len(grid_size) == 2:
        # Separate sizes for latitude and longitude
        lat_grid_size = grid_size[0]
        lng_grid_size = grid_size[1]
    else:
        # Same size for both
        lat_grid_size = grid_size
        lng_grid_size = grid_size
    
    # Calculate number of rows and columns in the grid
    lat_range = ne_corner[0] - sw_corner[0]
    lng_range = ne_corner[1] - sw_corner[1]
    
    num_rows = int(lat_range / lat_grid_size)
    num_cols = int(lng_range / lng_grid_size)
    
    # Create grid cell for each row and column
    for i in range(num_rows):
        for j in range(num_cols):
            lat_sw = sw_corner[0] + i * lat_grid_size
            lng_sw = sw_corner[1] + j * lng_grid_size
            
            lat_ne = lat_sw + lat_grid_size
            lng_ne = lng_sw + lng_grid_size
            
            # Calculate cell dimensions in km
            cell_width_km = calculate_distance_km(
                (lat_sw + lat_ne) / 2, lng_sw,
                (lat_sw + lat_ne) / 2, lng_ne
            )
            cell_height_km = calculate_distance_km(
                lat_sw, (lng_sw + lng_ne) / 2,
                lat_ne, (lng_sw + lng_ne) / 2
            )
            
            # Add rectangle for each grid cell with better styling
            folium.Rectangle(
                bounds=[[lat_sw, lng_sw], [lat_ne, lng_ne]],
                color='black',       # Black outline for better visibility
                weight=1,            # Thinner lines to avoid cluttering
                fill=False,          # No fill to see the map underneath
                opacity=0.7,         # Slightly transparent
                popup=f"Grid Cell ({i},{j})<br>ID: r{i}c{j}<br>SW: [{lat_sw:.6f}, {lng_sw:.6f}]<br>NE: [{lat_ne:.6f}, {lng_ne:.6f}]<br>Dimensions: {cell_width_km:.2f} x {cell_height_km:.2f} km<br>Area: {cell_width_km * cell_height_km:.2f} km²"
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
    
    # If predefined locations provided, add them as markers
    if predefined_locations:
        for name, location in predefined_locations.items():
            if "center" in location:
                folium.Marker(
                    location=location["center"],
                    popup=name,
                    tooltip=name
                ).add_to(m)
    
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

def calculate_polygon_area_km2(polygon_coords):
    """
    Calculate the area of a polygon in square kilometers using the shoelace formula
    and converting from degrees to kilometers
    
    Parameters:
    -----------
    polygon_coords : list
        List of [lat, lng] coordinate pairs defining the polygon
        
    Returns:
    --------
    float : Area in square kilometers
    """
    if len(polygon_coords) < 3:
        return 0
    
    # Convert to radians for more accurate calculation
    coords_rad = [(math.radians(lat), math.radians(lng)) for lat, lng in polygon_coords]
    
    # Earth radius in kilometers
    earth_radius = 6371
    
    # Calculate area using spherical excess formula for more accuracy
    # For small areas, we can use a simpler approach
    area = 0
    n = len(coords_rad)
    
    for i in range(n):
        j = (i + 1) % n
        lat1, lng1 = coords_rad[i]
        lat2, lng2 = coords_rad[j]
        
        # Spherical triangle area calculation
        area += lng2 * math.sin(lat1) - lng1 * math.sin(lat2)
    
    area = abs(area) * earth_radius * earth_radius / 2
    return area

def geocode_location(location_name, user_agent="uav_sensor_app"):
    """
    Geocode a location name to coordinates
    
    Parameters:
    -----------
    location_name : str
        Name of the location to geocode
    user_agent : str
        User agent string for the geocoder
        
    Returns:
    --------
    tuple or None
        (latitude, longitude) if successful, None otherwise
    """
    try:
        geolocator = Nominatim(user_agent=user_agent)
        location = geolocator.geocode(location_name)
        
        if location:
            return (location.latitude, location.longitude)
        else:
            return None
    except (GeocoderTimedOut, GeocoderUnavailable):
        # Handle geocoding errors gracefully
        return None
        
def reverse_geocode(lat, lng, user_agent="uav_sensor_app"):
    """
    Convert coordinates to a location name
    
    Parameters:
    -----------
    lat : float
        Latitude
    lng : float
        Longitude
    user_agent : str
        User agent string for the geocoder
        
    Returns:
    --------
    str or None
        Location name if successful, None otherwise
    """
    try:
        geolocator = Nominatim(user_agent=user_agent)
        location = geolocator.reverse((lat, lng))
        
        if location:
            return location.address
        else:
            return None
    except (GeocoderTimedOut, GeocoderUnavailable):
        # Handle geocoding errors gracefully
        return None