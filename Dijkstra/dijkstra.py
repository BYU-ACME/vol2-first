# dijkstra.py
"""Volume 2: Dijkstra's Algorithm
<Name>
<Class>
<Date>
"""

# imports
from math import sqrt, inf
from queue import PriorityQueue
from collections import defaultdict

class Edge:
    """An edge object, which wraps the node and weight attributes into one
    object, allowing for insertion/deletion from a set using just
    the node attribute

    Attributes:
        node (str): the value for the node the edge is pointing to
        weight (int): the weight of the edge
    """
    def __init__(self, node, weight):
        self.node = node
        self.weight = weight

    def __hash__(self):
        """Use only node attribute for hashing"""
        return hash(self.node)

    def __eq__(self, other):
        """Use only node attribute for equality"""
        if isinstance(other, Edge):
            return self.node == other.node
        return self.node == other

    def __str__(self):
        """String representation: a tuple-like view of the node and weight"""
        return f"({str(self.node)}, {str(self.weight)})"

    def __repr__(self):
        """Repr is used when edges are displayed in a set"""
        return f"Edge({repr(self.node)}, {repr(self.weight)})"

class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
        directed (bool): true if the graph is a directed graph.
    """
    def __init__(self, adjacency={}, directed=False):
        """Store the adjacency dictionary and directed as class attributes"""
        self.adjacency_d = dict(adjacency)

        # Store directed as an instance attribute
        self.directed = directed

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.adjacency_d)

    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        if n not in self.adjacency_d:
            self.adjacency_d[n] = set()

    def add_edge(self, u, v, weight=1.0):
        """Add a weighted edge between node u and node v.
        If an edge already exists between u and v, simply update the weight.
        Also add u and v to the graph if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
            weight: the edge's weight
        """
        # Add nodes (if they aren't already in the graph).
        self.add_node(u)
        self.add_node(v)

        # Remove old edge if present
        self.adjacency_d[u].discard(v)

        # Connect the nodes
        self.adjacency_d[u].add(Edge(v, weight))

        # Add reverse direction if undirected
        if not self.directed:

            self.adjacency_d[v].discard(u)
            self.adjacency_d[v].add(Edge(u, weight))

    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        # Remove the actual node (this raises the KeyError if needed).
        self.adjacency_d.pop(n)

        # Remove edges adjacent to the node.
        for neighbors in self.adjacency_d.values():
            neighbors.discard(n)

    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        self.adjacency_d[u].remove(v)     # u -/-> v (raises KeyError if needed).

        if not self.directed:
            self.adjacency_d[v].remove(u)     # v -/-> u (raises KeyError if needed).

    # Problem 1
    def shortest_path(self, source, target):
        """Begin Dijkstra's at the source node and proceed until the target is
        found. Return an integer denoting the sum of weights along the shortest
        path from source to target along with a list of the path itself,
        including endpoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            An int denoting the sum of weights along the shortest path
                from source to target
            A list of nodes along the shortest path from source to target,
                including the endpoints. The path should contain strings
                representing the nodes, not edge objects

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        raise NotImplementedError("Problem 1 Incomplete")

# Constants
FILE_PATH = r"bathymetry.tt3"
START_LONG_LAT = (131.75, -5.52)
END_LONG_LAT = (128.657, -3.576)
METERS_PER_DEG = 111111 # Approximation for meters per degree of lat/long
FAULT_RADIUS_METERS = 103587.2508/2 # Approximate radius for the fault plane, in meters
FAULT_PLANE_RADIUS = FAULT_RADIUS_METERS / METERS_PER_DEG  # radius of the fault plane approximated to degrees

class TsunamiModel(Graph):
    """A class representing a model for calculating time-based tsunami paths
    between two locations given the bathymetric (sea-depth) data of the area.

    Attributes:
        fault_plane_radius (float): Describes the radius (in arc-degrees)
        of the fault plane over which the tsunami forms, which is used to
        determine the proper starting point in time prediction.
        ncols, nrows (int): The number of columns and rows, respectively,
        in the bathymetry grid,
        lat_llcorner, long_llcorner (float) The longitude and latitude coordinates,
        respectively, of the lower-left corner of the bathymetry data grid.
        cellsize (float): The distance (in arc-degrees) between any two grid
        points in the bathymetric data.
        depths_grid (list[list[float]]): The grid of bathymetric data, each value
        represents the sea-depth at that location.
        long_lat_grid (list[list[float]]): The grid of latitude/longitude
        cooridnates corresponding to the locations of each bathymetric
        measurement in grid.
        shifted_start_long_lat (tuple): Indices corresponding to the starting point in grid
        of the tsunami.
        shifted_end_long_lat (tuple): Indices corresponding to the destination point in grid
        of the tsunami.

    Methods:
        calculate_tsunami_path(): Returns the estimated time (in minutes)
        and path for a tsunami to travel between shifted_start_long_lat
        and shifted_end_long_lat.
    """

    # Problem 2
    def __init__(self, filename, start_long_lat, end_long_lat,
                 fault_plane_radius=FAULT_PLANE_RADIUS):
        """
        Initializes the TimeModel.

        Parameters:
        - start_long_lat (tuple): Tuple containing start longitude and latitude
        which represents the tsunami's origin point.
        - end_long_lat (tuple): Tuple containing end longitude and latitude
        which represents the location in the tsunami's path.
        - fault_plane_radius (float): Describes the radius (in arc-degrees)
        of the fault plane over which the tsunami forms, which is used to
        determine the proper starting point in time prediction.

        Attributes:
        - ncols (int): number of columns in the depths matrix.
        - nrows (int): number of rows in the depths matrix.
        - fault_plane_radius (float): the radius of the earthquake's fault plate, 
        which is used to find the starting location of tsunami.
        - long_llcorner (float): the longitude of the lower left corner node
        - lat_llcorner (float): the latitude of the lower left corner node
        - cellsize (float): the arc-distance between two adjacent depth measurements,
        used to find the distance between two adjacent nodes.
        - depths_grid (list[list[float]]): (nrows, ncols) matrix of depth 
        measurements contained in bathymetry.tt3
        - long_lat_grid (list[list[float]]): (nrows, ncols) matrix of (longitude, latitude)
        coordinates for each depth measurement indexed the same as depths_grid
        - start_point (tuple): (longitude, latitude) start position of tsunami 
        shifted by the fault_plane_radius in direction of end_point
        - end_point (tuple): (longitude, latitude) target position of interest
        """
        super().__init__()
        # Read in bathymetry data from file
        self._read_file(filename)

        # Process longitude and latitude data
        self.fault_plane_radius = None

        raise NotImplementedError("Problem 2 Incomplete")

    def _read_file(self, filename):
        """
        Reads in the contents of the bathymetry datafile with filename
        and creates instance attributes with the relevant data.
        
        Parameters:
        - filename (str): filename to get bathymetry data
        """

        # Read in the bathymetry data
        with open(filename) as file:
            lines = file.readlines()

        # Extract grid properties (ncols, nrows, etc.) from the read lines
        self.ncols = int(lines[0])  # Extract the number of columns
        self.nrows = int(lines[1])  # Extract the number of rows
        self.long_llcorner = float(lines[2])  # Extract longitude-coordinate of lower left corner
        self.lat_llcorner = float(lines[3])  # Extract latitude-coordinate of lower left corner
        self.cellsize = float(lines[4])  # Extract cell size

        # Create a matrix/grid using data starting from line 6
        self.depths_grid = [list(map(int, line.split())) for line in lines[5:]]

    def _generate_long_lat_grid(self):
        """
        Generates a grid of longitude and latitude coordinates.

        Returns:
        - (list[list[float]]): Grid of longitude and latitude coordinates.
        """
        start_value = (self.long_llcorner, self.lat_llcorner)  # Starting longitude and latitude to line up with the bathymetry data
        increment = self.cellsize
        rows = self.nrows
        columns = self.ncols

        grid = [[(start_value[0] + col * increment,
                  start_value[1] + row * increment)
                 for col in range(columns)]
                 for row in range(rows)]

        return grid[::-1]  # Flipping the grid and returning it

    # Problem 2
    def _shifted_start(self, start_point, end_point):
        """
        Returns the shifted start location to line up with the edge of the fault
        plane in the direction of end_point.

        Parameters:
        - start_point (tuple): (longitude, latitude) of epicenter of the earthquake
        to be shifted to edge of the fault plane
        - end_point (tuple): (longitude, latitude) of target location. The start_point
        is shifted in direction of the end_point
        """
        raise NotImplementedError("Problem 2 Incomplete")

    # Problem 2
    def _long_and_lat(self, start_long_lat, end_long_lat):
        """
        Generates the instance attributes associated with latitude and
        longitude coordinates. Specifically, the starting and ending
        grid indices and the grid of longitude and latitude coordinates
        """
        self.long_lat_grid = None

        self.start_point = None
        self.end_point = None
        raise NotImplementedError("Problem 2 Incomplete")

    def _get_nearest_point(self, long_lat_point):
        """
        Finds the closest grid point to a given longitude and latitude point.

        Parameters:
        - long_lat_point (tuple): Latitude and longitude coordinates.

        Returns:
        - (tuple): Indices of the closest grid point.
        """
        closest_value_diff = float("inf")  # Initialize a variable to track the closest value difference
        closest_position = (0, 0) # Initialize a variable to store the closest position

        # Iterate through the grid to find the closest grid point
        for i in range(len(self.long_lat_grid)):
            for j in range(len(self.long_lat_grid[i])):
                current_tuple = self.long_lat_grid[i][j]  # Get current grid point coordinates

                # Calculate the absolute difference between the points
                value_diff = sum(abs(x - y) for x, y in zip(long_lat_point, current_tuple))

                # Check if the current grid point is closer than the previously closest one
                # and self.depths_grid[closest_position[0]][closest_position[1]] < 0
                thing = self.depths_grid[i][j]
                if value_diff < closest_value_diff and thing < 0:
                    closest_value_diff = value_diff  # Update the closest value difference
                    closest_position = (i, j)  # Update the closest position
        return closest_position   # Return the indices of the closest grid point

    # Problem 3
    def _get_neighbors(self, node):
        """
        Retrieves neighboring nodes of a given node within the depths grid boundaries.
        A neighboring node is valid if it is in bounds and has a negative elevation.

        Parameters:
        - node (tuple): Coordinates of the current node (row, col).

        Returns:
        - (list): List of neighboring nodes.
        """
        raise NotImplementedError("Problem 3 Incomplete")

    # Problem 3
    def _get_time(self, d1, d2):
        """
        Estimates the time (in seconds) for a tsunami to travel between two
        adjacent grid points with depths d1 and d2 respectively.

        Parameters:
        - d1 (float): depth of node 1
        - d2 (float): depth of node 2

        Attributes:
        - (float): the time to travel between two adjacent depths
        """
        raise NotImplementedError("Problem 3 Incomplete")
    
    # Problem 4
    def _convert_path_to_long_lat(self, path):
        """
        Converts a path of grid points to a list of longitude and latitude coordinates.

        Parameters:
        - path (list): List of grid points.

        Returns:
        - (list): List of longitude and latitude coordinates.
        """
        raise NotImplementedError("Problem 4 Incomplete")

    # Problem 4
    def _generate_graph(self):
        """
        Generates the graph used for Tsunami time prediction.
        Specifically, uses method of the Graph class to add edges
        between grid points below sea level, weighted by the get_time() method.
        Nodes are labeled by their zero-indexed location in grid.
        """
        raise NotImplementedError("Problem 4 Incomplete")

    # Problem 5
    def calculate_tsunami_path(self):
        """
        Implementation of Dijkstra's algorithm on a grid to find the shortest
        time for a tsunami to get from self.shifted_start_long_lat to self.shifted_end_long_lat

        Returns:
        - (float): Total time taken in minutes for the shortest path.
        - (list): Shortest path in longitude and latitude coordinates.
        """
        raise NotImplementedError("Problem 5 Incomplete")
