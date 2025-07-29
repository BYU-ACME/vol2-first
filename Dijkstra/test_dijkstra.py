"""Unit testing file for Dijkstra lab file"""

import pytest
from dijkstra import Edge, Graph, TsunamiModel

def test_shortest_path():
    """Test code for the shortest_path() method of the Graph class"""
    # Undirected graph
    graph = Graph({'A': {Edge('B', 1), Edge('D', 100)},
                   'B': {Edge('A', 1), Edge('C', 1), Edge('D', 100)},
                   'C': {Edge('B', 1), Edge('D', 1)},
                   'D': {Edge('A', 100), Edge('B', 100), Edge('C', 1)}})

    v1, t1 = graph.shortest_path('A', 'D')
    assert v1 == 3, "Incorrect weight sum"
    assert t1 == ['A', 'B', 'C', 'D'], "Incorrect path"

    graph.remove_edge('A', 'B')
    v2, t2 = graph.shortest_path('A', 'D')
    assert v2 == 100, "Incorrect weight sum"
    assert t2 == ['A', 'D'], "Incorrect path"

    with pytest.raises(KeyError):
        graph.shortest_path('A', 'Not a node')

    # Directed graph
    graph = Graph({'A': {Edge('D', 100)},
                   'B': {Edge('A', 1), Edge('D', 100)},
                   'C': {Edge('B', 1), Edge('D', 1)},
                   'D': {Edge('A', 100), Edge('B', 100), Edge('C', 1)}},
                  directed=True)

    v1, t1 = graph.shortest_path('A', 'D')
    assert v1 == 100, "Incorrect weight sum"
    assert t1 == ['A', 'D'], "Incorrect path"

    graph.add_edge('A', 'D', weight=1)
    v2, t2 = graph.shortest_path('A', 'B')
    assert v2 == 3, "Incorrect weight sum"
    assert t2 == ['A', 'D', 'C', 'B'], "Incorrect path"

def test_shifted_start():
    """Test code for the shifted_start() method of the TsunamiModel class"""

    FILE_PATH = r"bathymetry.tt3"
    START_LONG_LAT = (131.75, -5.52)
    END_LONG_LAT = (128.657, -3.576)
    prediction = TsunamiModel(FILE_PATH, START_LONG_LAT, END_LONG_LAT)

    assert prediction.shifted_start_long_lat == (166, 382),    "_shifted_start() failed, self.start_point should be equal to (166, 382)"
    assert prediction.shifted_end_long_lat == (65, 220),       "_long_and_lat() failed, ensure value of end_point is a point in the graph. Correct value is (65, 220)"

def test_generate_graph():
    """Test code for the _generate_graph() method of the TsunamiModel class"""

    FILE_PATH = r"bathymetry.tt3"
    START_LONG_LAT = (131.75, -5.52)
    END_LONG_LAT = (128.657, -3.576)
    prediction = TsunamiModel(FILE_PATH, START_LONG_LAT, END_LONG_LAT)
    prediction.calculate_tsunami_path()
    
    assert len(prediction.adjacency_d) == 212378, "_generate_graph() failed to create the correct number of nodes in the graph"