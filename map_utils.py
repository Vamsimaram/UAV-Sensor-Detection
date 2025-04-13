import folium
from folium.plugins import Draw

def create_grid_overlay(map_obj, sw_corner, ne_corner, grid_size):
    """
    Create a grid overlay on the map based on specified corners and grid size
    
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
            
            # Add rectangle for each grid cell with better styling
            folium.Rectangle(
                bounds=[[lat_sw, lng_sw], [lat_ne, lng_ne]],
                color='black',       # Black outline for better visibility
                weight=1,            # Thinner lines to avoid cluttering
                fill=False,          # No fill to see the map underneath
                opacity=0.7,         # Slightly transparent
                popup=f"Grid Cell ({i},{j})<br>ID: r{i}c{j}<br>SW: [{lat_sw:.6f}, {lng_sw:.6f}]<br>NE: [{lat_ne:.6f}, {lng_ne:.6f}]"
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
        <h4 style="margin-top: 0;">Drawing Instructions:</h4>
        <ul style="padding-left: 20px; margin-bottom: 0;">
            <li>Click the rectangle or polygon tool on the top right</li>
            <li>Draw your area on the map</li>
            <li>The coordinates will be captured automatically</li>
            <li>Click "Set Drawn Area" when finished</li>
            <li>Use the Download button below the map to save your boundary</li>
        </ul>
    </div>
    """
    
    # Add the instructions to the map
    instructions = folium.Element(instructions_html)
    m.get_root().html.add_child(instructions)
    
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