# type hint for classes, see https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

from types import FunctionType
import sys
import queue
import numpy as np
from pyvis.network import Network
import networkx as nx
import logging
logger = logging.getLogger(__name__)


class Node:
    """Representation of a single node in a graph."""

    def __init__(self, label: str = ""):
        """Constructor for a node, given label.

        Args:
            label (str, optional): Label for this node.
        """
        # A human readable name.
        self.label = label
        # The index of this node in a graph.
        self.index = -1

    def _setIndex(self, idx: int):
        """Sets the identication index within a graph. Used internally in graph functions.
        Don't set it manually.

        Args:
            idx (int): unique index within a graph.
        """
        self.index = idx

    def _getIndex(self) -> int:
        """Returns the index of this node within a graph. Used internally in graph functions.

        Returns:
            int: unique index within a graph.
        """
        return self.index

    def __str__(self) -> str:
        """Returns string representation of this node.

        Returns:
            str: Node as string: "index -> label". In "index" == "label", then only index is returned.
        """
        if (str(self.index) == self.label):
            return f"{self.index}"

        return f"{self.index} -> {self.label}"

    def __repr__(self) -> str:
        """Returns string representation of this node.

        Returns:
            str: Node as string: "index -> label".
        """
        return str(self)


class Graph:
    """Undirected Graph implementation. """

    def __init__(self):
        """Initializes internal representation. """
        # List of all graph's nodes.
        self.nodes = []
        # Edges between nodes stored in adjacency matrix.
        self.adjMatrix = []

    def get_nodes(self) -> list[Node]:
        """Returns all nodes.

        Returns:
            list: list of all nodes
        """
        return self.nodes

    def add_node(self, node: Node):
        """Add a node to the graph.

        Args:
            node (Node): new node to be added.
        """
        node._setIndex(len(self.nodes))
        self.nodes.append(node)

        # check if adjacency matrix is large enough
        if len(self.adjMatrix) < len(self.nodes):
            # ... nope, we need to increase the size
            # This is incredibly inefficient, but illustrative to show
            # what needs to happen when max size of graph is unclear
            # in advance and _no_ adjacency lists shall be used.
            # logger.debug("Increasing size of adjMatrix.")
            newAdjMatrix = []
            for i in range(len(self.nodes)):
                newAdjMatrix.append([0 for i in range(len(self.nodes))])

            # Copy old elements to new
            for i in range(len(self.adjMatrix)):
                for j in range(len(self.adjMatrix)):
                    newAdjMatrix[i][j] = self.adjMatrix[i][j]

            self.adjMatrix = newAdjMatrix

    def toNumpyArray(self) -> np.ndarray:
        """Transforms internal adjacency matrix to a numpy.array.

        Returns:
            np.array: adjacency matrix as numpy array.
        """
        return np.array(self.adjMatrix)

    def getLabelAdjacencyList(self, nodeID: int) -> list:
        """Transforms internal adjacency matrix to adjacency list for a given node,
        using the nodes labels.

        Args:
            nodeID (int): Node index for which adjacency list shall be returned.

        Returns:
            list: list of all connected nodes, given by their label.
        """
        return [self.nodes[i].label for i, n in enumerate(self.adjMatrix[nodeID]) if n != 0]

    def getIndexAdjacencyList(self, nodeID: int) -> list[int]:
        """Transforms internal adjacency matrix to adjacency list for a given node,
        using the nodes indices.

        Args:
            nodeID (int): Node index for which adjacency list shall be returned.

        Returns:
            list: list of all connected nodes, given by their indices.
        """
        return [i for i, n in enumerate(self.adjMatrix[nodeID]) if n != 0]

    def getNodeAdjacencyList(self, node: Node) -> list[Node]:
        """Transforms internal adjacency matrix to adjacency list for a given node,
        using the nodes object itself.

        Args:
            node (Node): Node object for which adjacency list shall be returned.

        Returns:
            list: list of all connected nodes, given by their objects.
        """
        logger.debug(f"Getting neighbors for {node}")
        indicesList = self.getIndexAdjacencyList(node._getIndex())
        logger.debug(f"   -> {indicesList}")
        return [self.get_node_by_index(i) for i in indicesList]

    def getLabelGraphAdjacencyList(self) -> dict[str, list[Node]]:
        """Return the adjacency list for all nodes of this graph,
        using the label of each node as identifier in the dict.

        Returns:
            dict: adjacency list for each node, for given node i: {label(i): ...}
        """
        return {node.label: self.getLabelAdjacencyList(i) for (i, node) in enumerate(self.nodes)}

    def getIndexGraphAdjacencyList(self) -> dict[int, list[int]]:
        """Return the adjacency list for all nodes of this graph,
        using the index of each node as identifier in the dict.

        Returns:
            dict: adjacency list for each node, for given node i: {index(i): ...}
        """
        return {i: self.getIndexAdjacencyList(i) for (i, node) in enumerate(self.nodes)}

    def getNodeGraphAdjacencyList(self) -> dict[Node, list[Node]]:
        """Return the adjacency list for all nodes of this graph,
        using the node itself as identifier in the dict.

        Returns:
            dict: adjacency list for each node, for given node i: {i: ...}
        """
        return {node: self.getLabelAdjacencyList(i) for (i, node) in enumerate(self.nodes)}

    def toNetworkx(self, returnObject: nx.Graph = nx.Graph()) -> nx.Graph:
        """Transforms this graph instance into a Graph from the networkx package. Used for visualization.

        Args:
            returnObject (Node): Node object for which adjacency list shall be returned.

        Returns:
            nx.Graph: nx.Graph object or derived class, e.g. nx.DiGraph
        """
        G = returnObject
        # Add nodes
        for i, node in enumerate(self.nodes):
            logger.debug(f"toNetworkx - adding node: {node.label}")
            G.add_node(node.label, data={"label": node.label})
        # Add edges
        for i, row in enumerate(self.adjMatrix):
            for j, weight in enumerate(row):
                if (weight >= 1):
                    logger.debug(
                        f"toNetworkx - adding edge: {i}->{j} ({self.nodes[i].label}->{self.nodes[j].label})")
                    G.add_edge(self.nodes[i].label,
                               self.nodes[j].label, label=str(weight), weight=weight)
        return G

    def getEdgeLabels(self) -> dict[tuple[str, str], str]:
        """Returns labels for all existing edges in graph.
        The edge labels are retrieved from the adjacency matrix and values other than "1" (= edge exists)
        can be added to an edge when add_edge() is called with the weight argument !

        Returns:
            dict: Edge labels in form of dict. When edge exists between i and j,
            then dict looks like: {(label(i), label(j)): edge_label}
        """
        edge_labels = dict()
        for i in range(len(self.adjMatrix)):
            for j in range(len(self.adjMatrix)):
                if (self.adjMatrix[i][j] != 0):
                    # logger.debug(f"Edge label: {i}->{j}: {self.adjMatrix[i][j]}")
                    edge_labels[(self.nodes[i].label,
                                 self.nodes[j].label)] = self.adjMatrix[i][j]
        return edge_labels

    def add_edge_between(self, label1: str, label2: str, weight=1):
        """Add an edge between two nodes, given by the <em>labels</em> of these nodes.

        Args:
            label1 (str): label of the first connecting node.
            label2 (str): label of the second connecting node.
            weight (int, optional): Optional weight of edge. Defaults to 1 == edge exists. Used to label edges in getEdgeLabels().
        """
        n1 = self.get_node_by_label(label1)
        n2 = self.get_node_by_label(label2)
        self.add_edge(n1._getIndex(), n2._getIndex(), weight)

    def add_edge(self, v1: int, v2: int, weight=1):
        """Add an edge between two nodes, given by the <em>index</em> of these nodes in adjacency matrix.

        Args:
            v1 (int): index of the first connecting node.
            v2 (int): index of the second connecting node.
            weight (int, optional): Optional weight of edge. Defaults to 1 == edge exists. Used to label edges in getEdgeLabels().

        Raises:
            ValueError: when v1 == v2, no self-references allowed.
        """
        if v1 == v2:
            raise ValueError(
                f"Graph does not support self-references. {v1}")
        logger.debug(f"Adding edge between {v1} and {v2}")
        self.adjMatrix[v1][v2] = weight
        self.adjMatrix[v2][v1] = weight

    def get_edges(self, sortedReturn=False) -> dict[tuple[Node, Node], int]:
        """
        Return all edges with their weight.

        Returning dict has tuple of connected nodes as key, the edge weight as value, e.g.:
        <tt>
        {
            (A, B): 2,
            (C, D): 4,
            (A, D): 3
        }
        </tt>

        Args:
            sortedReturn (bool, optional): Sort edges by their weight, ascending.

        Returns:
            dict: edges in structure of type {(nodeI, nodeJ): weight}, see description above.
        """
        edges = dict()
        for i, row in enumerate(self.adjMatrix):
            for j, weight in enumerate(row):
                # Only consider "upper half" of adjacency matrix => undirected graph without self-references
                if i < j:
                    if weight != 0:
                        nodeI = self.get_node_by_index(i)
                        nodeJ = self.get_node_by_index(j)
                        edges[(nodeI, nodeJ)] = weight

        if (sortedReturn):
            return dict(sorted(edges.items(), key=lambda item: item[1]))

        return edges

    def get_node_by_index(self, index: int) -> Node:
        """Returns a node from the graph with the given index.

        Args:
            index (int): Index of the node to be retrieved.

        Returns:
            Node: The corresponding node object.
        """
        # logger.debug(f"Getting node by index: {index}")
        for n in self.nodes:
            if n.index == index:
                return n
        print(f"Current nodes: {self.nodes}")
        raise ValueError(f"Index {index} does not exist in graph.")
        # Hmm, this results in an endless loop?!
        # return next(n for n in self.nodes if n.index == index)

    def get_node_by_label(self, label: str) -> Node:
        """Returns a node from the graph with the given label.

        Args:
            label (int): Label of the node to be retrieved.

        Returns:
            Node: The corresponding node object.
        """
        if label == "":
            raise ValueError(f"Label must not be empty.")
        for n in self.nodes:
            if n.label == label:
                return n
        raise ValueError(
            f"Node with label {label} does not exist in graph: {self.nodes}")

    def get_edge_weight(self, v1: int, v2: int) -> int:
        """Returns the edge weight between two nodes.

        Args:
            v1 (int): index of the first connecting node.
            v2 (int): index of the second connecting node.

        Returns:
            int: weight of the edge between the given nodes
        """
        return self.adjMatrix[v1][v2]

    def remove_edge(self, v1: int, v2: int):
        """Removes an edge from the graph.

        Args:
            v1 (int): index of the first connecting node.
            v2 (int): index of the second connecting node.

        Raises:
            ValueError: when no edge between v1 and v2 exists.
        """
        if self.adjMatrix[v1][v2] == 0:
            raise ValueError(
                f"No edge between {v1} and {v2} which could be deleted.")
        self.adjMatrix[v1][v2] = 0
        self.adjMatrix[v2][v1] = 0

    def get_previous_nodes(self, v1: int) -> list[int]:
        """Get "parents" of this node, meaning the nodes having a connection to given <em>v1</em>.

        The parents can be retrieved by looking at the "column" with index <em>v1</em> of the adjacency table.

        Args:
            v1 (int): The index of the node for which its parents should be returned.

        Returns:
            list[int]: List of index with parents nodes.
        """
        # Solution without list comprehension
        # previous_nodes = []
        # for i, row in enumerate(self.adjMatrix):
        #    for j, weight in enumerate(row):
        #        # logger.debug(f"  checking [{i},{j}] | {j} == {v1}?")
        #        if j == v1 and weight > 0:
        #            previous_nodes.append(i)
        # return previous_nodes
        return [i for i, row in enumerate(self.adjMatrix) for j, weight in enumerate(row) if j == v1 and weight > 0]

    def traversal_BFS(self, startingNodeLabel: str, nodeFunction: FunctionType | None = None) -> list[Node]:
        """Breadth-first-search for traversing nodes in graph.

        Args:
            startingNodeLabel (str): the starting point, given by the label of the node.
            nodeFunction (FunctionType, optional): if not None, this function will be called for each node.

        Raises:
            ValueError: when starting node not found

        Returns:
            list: list of all nodes in order of BFS.
        """
        logger.debug(f"traversal_BFS - {startingNodeLabel}.")

        initialNode = self.get_node_by_label(startingNodeLabel)
        if (initialNode == None):
            raise ValueError(f"Node with label {startingNodeLabel} not found")

        returnList = list()
        nodeQueue = queue.Queue()
        visitedList = list()
        nodeQueue.put(initialNode)
        visitedList.append(initialNode)

        # as long as we have still nodes to go ...
        while (not nodeQueue.empty()):
            logger.debug(f"  Current queue - {list(nodeQueue.queue)}.")

            # ... dequeue next from queue
            curNode = nodeQueue.get()
            logger.debug(f"  Working on - {curNode}.")

            # Do something with curNode if function is given.
            if nodeFunction != None:
                nodeFunction(curNode)
            # here: Just add to returnList
            returnList.append(curNode)

            # Get all connected nodes
            connectedNodes = self.getNodeAdjacencyList(curNode)
            for node in connectedNodes:
                # if node not already discovered before ...
                if (node in visitedList):
                    logger.debug(f"    been here before - {node.label}.")
                else:
                    # ... mark as "discovered" and add to queue
                    visitedList.append(node)
                    nodeQueue.put(node)

        return returnList

    def traversal_DFS(self, startingNodeLabel: str, nodeFunction: FunctionType | None = None) -> list[Node]:
        """Depth-first-search for traversing nodes in graph.

        Args:
            startingNodeLabel (str): the starting point, given by the label of the node.
            nodeFunction (FunctionType, optional): if not None, this function will be called for each node.

        Raises:
            ValueError: when starting node not found

        Returns:
            list: list of all nodes in order of DFS.
        """
        logger.debug(f"traversal_DFS - {startingNodeLabel}.")

        initialNode = self.get_node_by_label(startingNodeLabel)
        if (initialNode == None):
            raise ValueError(f"Node with label {startingNodeLabel} not found")

        returnList = list()
        nodeStack = list()
        nodeStack.append(initialNode)
        visitedList = list()
        visitedList.append(initialNode)

        # as long as we have still nodes to go ...
        while (len(nodeStack) > 0):
            logger.debug(f"  Current stack - {nodeStack}.")

            # ... pop next from queue
            curNode = nodeStack.pop()
            logger.debug(f"  Working on - {curNode}.")

            # Do something with curNode if function is given.
            if nodeFunction != None:
                nodeFunction(curNode)
            # here: Just add to returnList
            returnList.append(curNode)

            # Get all connected nodes
            connectedNodes = self.getNodeAdjacencyList(curNode)
            for node in connectedNodes:
                # if node not already discovered before ...
                if (node in visitedList):
                    logger.debug(f"    been here before - {node.label}.")
                else:
                    # ... mark as "discovered" and add to stack
                    visitedList.append(node)
                    nodeStack.append(node)

        return returnList

    def shortestPaths_Dijkstra(self, startingNodeLabel: str) -> tuple[dict[Node, Node], dict[Node, int]]:
        """Implementation of Dijkstra's shortest path algorithm.
        This will calculate the shortest path to <em>all</em> other nodes in the graph.

        Function returns a tuple with two dicts:
            - previous_nodes: dict with the best previous node - this indicates
              the way from the destination to the start by following the previous nodes.
              dict[current, parent] -> dict[parent, grandparent] ...
            - shortest_path: dict which contains the distance to every other node in the graph.
              dict[Node, int], where for every Node of the Graph int is the distance to startingNodeLabel.

        Args:
            startingNodeLabel (str): label of the node to start from.

        Raises:
            ValueError: when starting label not found

        Returns:
            tuple[dict, dict]: (previous_nodes, shortest_path), see description above.
        """
        logger.debug(f"shortestPath_Dijkstra - {startingNodeLabel}.")

        startNode = self.get_node_by_label(startingNodeLabel)
        if (startNode == None):
            raise ValueError(f"Node with label {startingNodeLabel} not found")

        unvisited_nodes = self.get_nodes().copy()  # Shallow copy!!
        max_value = sys.maxsize  # == "infinity"

        # Initial shortest path with infinity except for start node
        shortest_path = {n: max_value for n in unvisited_nodes}
        shortest_path[startNode] = 0

        previous_nodes = {}
        count = 0

        while unvisited_nodes:
            count += 1
            logger.debug(
                f"{count}. graph iteration. unvisited_nodes: {unvisited_nodes}")

            # Next node to be considered: the one with minimum distance
            # Find the node with the minimum distance ... no priority queue here
            current_min_node = None
            for node in unvisited_nodes:
                if current_min_node == None:
                    current_min_node = node
                elif shortest_path[node] < shortest_path[current_min_node]:
                    current_min_node = node
            logger.debug(f"  current_min_node: {current_min_node}")

            if (current_min_node != None):
                # get neighbors and update their distance
                neighbors = self.getNodeAdjacencyList(current_min_node)
                for neighbor in neighbors:
                    if neighbor:
                        # path to this node is sum to current plus distance to neighbor
                        tmp_distance = shortest_path[current_min_node] + self.get_edge_weight(
                            current_min_node._getIndex(), neighbor._getIndex())
                        # is the path to this neighbor shorter?
                        if tmp_distance < shortest_path[neighbor]:
                            # update path length to better one
                            shortest_path[neighbor] = tmp_distance
                            # We also update the best path to the current node
                            previous_nodes[neighbor] = current_min_node

                logger.debug(f"  Shortest paths: {shortest_path}")
                logger.debug(f"  Previous nodes: {previous_nodes}")

                # current node is done
                unvisited_nodes.remove(current_min_node)

        return previous_nodes, shortest_path

    def shortestPaths_Dijkstra_pqueue(self, startingNodeLabel: str) -> tuple[dict[Node, Node], dict[Node, int]]:
        """Implementation of Dijkstra's shortest path algorithm - as in shortestPaths_Dijkstra(),
        but internally with a queue.PriorityQueue.

        This will calculate the shortest path to <em>all</em> other nodes in the graph.

        Function returns a tuple with two dicts:
            - previous_nodes: dict with the best previous node - this indicates
              the way from the destination to the start by following the previous nodes.
              dict[current, parent] -> dict[parent, grandparent] ...
            - shortest_path: dict which contains the distance to every other node in the graph.
              dict[Node, int], where for every Node of the Graph int is the distance to startingNodeLabel.

        Args:
            startingNodeLabel (str): label of the node to start from.

        Raises:
            ValueError: when starting label not found

        Returns:
            tuple[dict, dict]: (previous_nodes, shortest_path), see description above.
        """
        startNode = self.get_node_by_label(startingNodeLabel)
        if (startNode == None):
            raise ValueError(f"Node with label {startingNodeLabel} not found")

        visited = set()
        unvisited_nodes = queue.PriorityQueue()
        unvisited_nodes.put((0, startNode))
        max_value = sys.maxsize  # == "infinity"

        # Initial shortest path with infinity except for start node
        shortest_path = {n: max_value for n in self.nodes}
        shortest_path[startNode] = 0

        previous_nodes = {}
        count = 0

        while unvisited_nodes:
            count += 1
            logger.debug(
                f"{count}. graph iteration. unvisited_nodes: {unvisited_nodes}")

            logger.debug(f"  Prio Queue: {list(unvisited_nodes.queue)}")
            # get "nearest" node from priority queue
            while not unvisited_nodes.empty():
                _, current_min_node = unvisited_nodes.get()
                # loop until we get a not known node
                if current_min_node not in visited:
                    break
            else:
                break

            logger.debug(f"  current min node: {current_min_node}")
            # get neighbors and update their distance
            neighbors = self.getNodeAdjacencyList(current_min_node)
            for neighbor in neighbors:
                if neighbor:
                    # path to this node is sum to current plus distance to neighbor
                    tmp_distance = shortest_path[current_min_node] + self.get_edge_weight(
                        current_min_node._getIndex(), neighbor._getIndex())
                    # is the path to this neighbor shorter?
                    if tmp_distance < shortest_path[neighbor]:
                        # Add node to priority queue with the new distance
                        unvisited_nodes.put((tmp_distance, neighbor))
                        # update path length to better one
                        shortest_path[neighbor] = tmp_distance
                        # We also update the best path to the current node
                        previous_nodes[neighbor] = current_min_node

            logger.debug(f"  Shortest paths: {shortest_path}")
            logger.debug(f"  Previous nodes: {previous_nodes}")

        return previous_nodes, shortest_path

    def make_path(self, previous_nodes: dict, shortestPaths: dict, startingNodeLabel: str, endNodeLabel: str) -> list[Node]:
        """Helper function: returns the shortest path from startingNodeLabel to the destination endNodeLabel of a graph as a list of nodes.

        Args:
            previous_nodes (dict): dict with previous nodes, as returned from shortestPaths_Dijkstra()
            shortestPaths (dict): dict with shortestPaths, as returned from shortestPaths_Dijkstra()
            startingNodeLabel (str): node to start the path from
            endNodeLabel (str): node to end the path at

        Returns:
            list: list of nodes giving the way from startingNodeLabel to endNodeLabel
        """
        startNode = self.get_node_by_label(startingNodeLabel)
        endNode = self.get_node_by_label(endNodeLabel)
        currentNode = self.get_node_by_label(endNodeLabel)

        pathList = []

        while (currentNode != startNode):
            pathList.append(currentNode)
            # get previous node for current
            currentNode = previous_nodes[currentNode]

        pathList.append(startNode)
        return pathList[::-1]

    def _is_cyclic(self, curNode: Node, visited: list[Node], parent: Node | None) -> bool:
        """Internal helper function to recursively check if a(n undirected) graph contains a cycle.
        For an undirected graph we need to exclude the "parent", otherwise two connected nodes
        (0,1)<->(1,0) will always indicate to have a cyclic connection.

        As user, better call is_cycle_free().

        Args:
            curNode (Node): current node to be considered in this recursive call.
            visited (list[Node]): a list of all nodes we have seen before.
            parent (Node): the parent node to be ignored when checking for cycles.

        Returns:
            bool: True if there is a cycle in the graph.
        """
        logger.debug(f"  _is_cyclic for {curNode}")
        visited.append(curNode)

        # Check for all connected nodes ...
        for i in self.getNodeAdjacencyList(curNode):
            # ... when we haven't seen them before
            if (i not in visited):
                logger.debug(f"  Not yet visited: {i}")
                # ... call recursively with the neighbor node and current as parent
                if (self._is_cyclic(i, visited, curNode)):
                    logger.debug(f"  is_cyclic is True for: {i}")
                    return True

            # ... when we have seen them
            elif (parent != i):
                # ... and this node is not the parent we want to ignore, we have a cycle
                logger.debug(f"  Parent: {parent}")
                return True

        # No cycle found
        return False

    def is_cycle_free(self, startIndex: int = 0) -> bool:
        """
        Checks graph for cycles.

        Implementation is similar to Depth First Traversal.
        Once we detect an node we have been before, this will return False.

        Utilizes _is_cyclic() helper function.

        Args:
            startIndex (int, optional): hint on which node to start from.
            There must be edges to that node, otherwise DFS traversal will not work!!

        Returns:
            bool: True if no cycles exists, False is cycles exists.
        """
        if len(self) in (0, 1, 2):
            return True

        visited = []

        # For all nodes, we check with _is_cyclic for cycles
        for n in self.get_nodes():
            if n not in visited:
                if (self._is_cyclic(n, visited, None)) == True:
                    return False

        return True

    def mst_Kruskal(self) -> Graph:
        """
        Implementation of Kruskal's Algorithm to calculate the Minimal Spanning Tree to this graph.

        Returns:
            Graph: sub-graph with connections matching MST criteria.
        """
        logger.debug(f"mst_Kruskal")

        # Initialize minimal spanning tree (= graph without cycles) with copy of all nodes
        resultMST = Graph()
        for node in self.get_nodes():
            resultMST.add_node(node)

        listOfEdges = self.get_edges(sortedReturn=True)
        logger.debug(f"  listOfEdges: {listOfEdges}")

        # Take "lowest cost" edge and add to MST
        for nodes, weight in listOfEdges.items():
            logger.debug(f"  Considering edge {nodes}")
            resultMST.add_edge(nodes[0]._getIndex(),
                               nodes[1]._getIndex(), weight)

            # If we now added a cycle to the graph, remove this edge!!
            if (not resultMST.is_cycle_free(nodes[0]._getIndex())):
                logger.debug(f"  Introduced cycle ... remove it again.")
                resultMST.remove_edge(
                    nodes[0]._getIndex(), nodes[1]._getIndex())

            logger.debug(
                f"  Edges at end of iteration {resultMST.get_edges()}")

        return resultMST

    def __len__(self) -> int:
        """Returns node count.

        Returns:
            int: number of nodes in graph.
        """
        return len(self.nodes)

    def print_matrix(self):
        """Prints adjacency matrix to console."""

        for i, row in enumerate(self.adjMatrix):
            # first row header
            if (i == 0):
                print(f'       ', end='')
                for j, val in enumerate(row):
                    print(f'{j:4}', end='')
                print()
                print(f'        ', end='')
                for j, val in enumerate(row):
                    print(f'----', end='')
                print()

            print(f'{i:4} | ', end='')

            for j, val in enumerate(row):
                print(f'{val:4}', end='')

            print()


if __name__ == '__main__':
    """main guard for module."""
    print("No main function in here.")
