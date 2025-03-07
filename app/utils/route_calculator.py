import math
import heapq
from geopy.distance import geodesic

def calculate_optimal_route(start, end, blockages):
    """
    Calculate the optimal route using Dijkstra's algorithm, avoiding road blockages
    
    Args:
        start (dict): Start location with 'lat' and 'lng' keys
        end (dict): End location with 'lat' and 'lng' keys
        blockages (list): List of Post objects representing road blockages
        
    Returns:
        dict: Route information including waypoints, distance, and estimated time
    """
    # Convert blockages to a list of points to avoid
    blockage_points = []
    for blockage in blockages:
        # Create a penalty radius around each blockage based on estimated blockage time
        # The more severe the blockage, the larger the radius to avoid
        radius = min(0.01 * (blockage.estimated_blockage_time / 60), 0.05)  # Max 0.05 degrees (roughly 5km)
        blockage_points.append({
            'lat': blockage.latitude,
            'lng': blockage.longitude,
            'radius': radius,
            'penalty': blockage.estimated_blockage_time  # Time penalty in minutes
        })
    
    # Create a grid of points for the route calculation
    # In a real implementation, you would use actual road network data
    # For this demo, we'll create a simple grid between start and end
    grid_points = create_grid(start, end, blockage_points)
    
    # Calculate the shortest path using Dijkstra's algorithm
    path = dijkstra_shortest_path(grid_points, start, end)
    
    # Calculate total distance and estimated time
    total_distance = 0
    total_time = 0
    
    for i in range(len(path) - 1):
        point1 = (path[i]['lat'], path[i]['lng'])
        point2 = (path[i + 1]['lat'], path[i + 1]['lng'])
        distance = geodesic(point1, point2).kilometers
        total_distance += distance
        
        # Estimate time based on average speed of 40 km/h
        time_minutes = (distance / 40) * 60
        total_time += time_minutes
    
    return {
        'waypoints': path,
        'total_distance': round(total_distance, 2),
        'estimated_time': round(total_time, 0)
    }

def create_grid(start, end, blockage_points, grid_density=10):
    """
    Create a grid of points between start and end for route calculation
    
    Args:
        start (dict): Start location with 'lat' and 'lng' keys
        end (dict): End location with 'lat' and 'lng' keys
        blockage_points (list): List of blockage points to avoid
        grid_density (int): Number of points in each direction
        
    Returns:
        list: List of grid points with neighbors
    """
    # Calculate the bounding box
    min_lat = min(start['lat'], end['lat'])
    max_lat = max(start['lat'], end['lat'])
    min_lng = min(start['lng'], end['lng'])
    max_lng = max(start['lng'], end['lng'])
    
    # Add some padding to the bounding box
    padding = 0.02  # About 2km
    min_lat -= padding
    max_lat += padding
    min_lng -= padding
    max_lng += padding
    
    # Create a grid of points
    lat_step = (max_lat - min_lat) / grid_density
    lng_step = (max_lng - min_lng) / grid_density
    
    grid_points = []
    
    # Add start and end points to the grid
    grid_points.append({
        'id': 'start',
        'lat': start['lat'],
        'lng': start['lng'],
        'neighbors': []
    })
    
    grid_points.append({
        'id': 'end',
        'lat': end['lat'],
        'lng': end['lng'],
        'neighbors': []
    })
    
    # Create grid points
    for i in range(grid_density + 1):
        for j in range(grid_density + 1):
            lat = min_lat + i * lat_step
            lng = min_lng + j * lng_step
            
            # Skip if too close to start or end
            if (geodesic((lat, lng), (start['lat'], start['lng'])).kilometers < 0.1 or
                geodesic((lat, lng), (end['lat'], end['lng'])).kilometers < 0.1):
                continue
            
            grid_points.append({
                'id': f'point_{i}_{j}',
                'lat': lat,
                'lng': lng,
                'neighbors': []
            })
    
    # Connect points to their neighbors
    for i, point in enumerate(grid_points):
        for j, other_point in enumerate(grid_points):
            if i == j:
                continue
            
            # Calculate distance between points
            distance = geodesic(
                (point['lat'], point['lng']), 
                (other_point['lat'], other_point['lng'])
            ).kilometers
            
            # Only connect if within a reasonable distance
            if distance < 5:  # 5km max connection distance
                # Check if the connection passes through a blockage
                penalty = calculate_blockage_penalty(
                    point, other_point, blockage_points
                )
                
                # Add as neighbor with distance and penalty
                point['neighbors'].append({
                    'id': other_point['id'],
                    'distance': distance,
                    'penalty': penalty
                })
    
    return grid_points

def calculate_blockage_penalty(point1, point2, blockage_points):
    """
    Calculate penalty for a path segment based on proximity to blockages
    
    Args:
        point1 (dict): First point with 'lat' and 'lng' keys
        point2 (dict): Second point with 'lat' and 'lng' keys
        blockage_points (list): List of blockage points to avoid
        
    Returns:
        float: Penalty value (minutes to add to travel time)
    """
    total_penalty = 0
    
    for blockage in blockage_points:
        # Check if the line segment passes near the blockage
        distance = point_to_line_distance(
            blockage['lat'], blockage['lng'],
            point1['lat'], point1['lng'],
            point2['lat'], point2['lng']
        )
        
        # If within the blockage radius, apply a penalty
        if distance < blockage['radius']:
            # Penalty increases as distance decreases
            proximity_factor = 1 - (distance / blockage['radius'])
            penalty = blockage['penalty'] * proximity_factor
            total_penalty += penalty
    
    return total_penalty

def point_to_line_distance(px, py, x1, y1, x2, y2):
    """
    Calculate the shortest distance from a point to a line segment
    
    Args:
        px, py: Point coordinates
        x1, y1: Line segment start coordinates
        x2, y2: Line segment end coordinates
        
    Returns:
        float: Distance in coordinate units (degrees)
    """
    # Convert to kilometers for more accurate distance calculation
    p = (px, py)
    start = (x1, y1)
    end = (x2, y2)
    
    # Calculate distances
    d_start = geodesic(p, start).kilometers
    d_end = geodesic(p, end).kilometers
    d_line = geodesic(start, end).kilometers
    
    # If the line segment is very short, just return the distance to either end
    if d_line < 0.001:
        return min(d_start, d_end)
    
    # Calculate the projection of the point onto the line
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (d_line ** 2)))
    
    # Calculate the closest point on the line
    closest_x = x1 + t * (x2 - x1)
    closest_y = y1 + t * (y2 - y1)
    
    # Return the distance to the closest point
    return geodesic((px, py), (closest_x, closest_y)).kilometers

def dijkstra_shortest_path(grid_points, start, end):
    """
    Find the shortest path using Dijkstra's algorithm
    
    Args:
        grid_points (list): List of grid points with neighbors
        start (dict): Start location with 'lat' and 'lng' keys
        end (dict): End location with 'lat' and 'lng' keys
        
    Returns:
        list: List of waypoints forming the shortest path
    """
    # Create a dictionary of points by ID for easy lookup
    points_by_id = {point['id']: point for point in grid_points}
    
    # Initialize distances
    distances = {point['id']: float('infinity') for point in grid_points}
    distances['start'] = 0
    
    # Initialize previous nodes for path reconstruction
    previous = {point['id']: None for point in grid_points}
    
    # Priority queue for Dijkstra's algorithm
    queue = [(0, 'start')]
    
    # Set of visited nodes
    visited = set()
    
    while queue:
        current_distance, current_id = heapq.heappop(queue)
        
        # If we've reached the end, we're done
        if current_id == 'end':
            break
        
        # Skip if already visited
        if current_id in visited:
            continue
        
        visited.add(current_id)
        
        # Get the current point
        current_point = points_by_id[current_id]
        
        # Check all neighbors
        for neighbor in current_point['neighbors']:
            neighbor_id = neighbor['id']
            
            # Skip if already visited
            if neighbor_id in visited:
                continue
            
            # Calculate distance including penalty
            distance = current_distance + neighbor['distance']
            time_penalty = neighbor['penalty']
            
            # Convert time penalty to equivalent distance
            # Assuming average speed of 40 km/h, 1 minute = 40/60 = 2/3 km
            distance_penalty = time_penalty * (2/3)
            total_cost = distance + distance_penalty
            
            # Update distance if shorter path found
            if total_cost < distances[neighbor_id]:
                distances[neighbor_id] = total_cost
                previous[neighbor_id] = current_id
                heapq.heappush(queue, (total_cost, neighbor_id))
    
    # Reconstruct the path
    path = []
    current_id = 'end'
    
    while current_id:
        point = points_by_id[current_id]
        path.append({
            'lat': point['lat'],
            'lng': point['lng']
        })
        current_id = previous[current_id]
    
    # Reverse the path to start->end order
    path.reverse()
    
    return path
