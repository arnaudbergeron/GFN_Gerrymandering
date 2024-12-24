import torch

from utils.data_utils import load_raw_data


class Graph:
    def __init__(self, device):
        # Dictionary to store vertices on the border and their corresponding district ids
        self.vertex_on_border = {} # dict of (vertex_id, district_id) -> tensor representing a possible action not sure tensor is necessary
        self.district_id_on_border_of_vertex = {} # dict of vertex_id -> set of bodering district_ids
        self.vertices = {}
        self.device = device
        self.max_district_id = 0
        self.state = None  # Tensor representing the current state of the graph
        self.taken_actions = None
        self.num_county = 0
        self.previous_state = None
        self.df = None
        # self.state[vertex_id, 0] is the district_id of the vertex with vertex_id negative if it is on the border
        # self.state[vertex_id, 1:] is the adjacent vertices of the vertex with vertex_id

    def graph_from_json(self, path_to_json):
        """Creates a graph from a JSON data file.

        Args:
            path_to_json (str): The path to the JSON data file.
        """
        df = load_raw_data(path_to_json)
        max_adjacent_vertices = 0

        num_districts = df['cd_2020'].nunique()
        self.max_district_id = num_districts
        self.num_county = df.shape[0]
        self.taken_actions = torch.zeros((df.shape[0], num_districts), device=self.device)
        for index, row in df.iterrows():
            prescinct_id = index
            district_id = row['cd_2020']

            self.taken_actions[prescinct_id, district_id-1] = 1
            
            vertex = Vertex(prescinct_id=prescinct_id, district_id=district_id)
            vertex.set_adj(row['adj'])

            # Update the maximum number of adjacent vertices
            if len(row['adj']) > max_adjacent_vertices:
                max_adjacent_vertices = len(row['adj'])

            self.vertices[prescinct_id] = vertex


        # Initialize the state tensor with zeros
        self.state = torch.zeros((len(self.vertices), max_adjacent_vertices + 1 + num_districts), device=self.device)
        self.previous_state = torch.zeros((len(self.vertices)), device=self.device)

        # Populate the state tensor with district ids and adjacent vertices
        for vertex_id, vertex in self.vertices.items():
            self.state[vertex_id, 0] = vertex.district_id

            # we represent empty adj vertices with 0
            _adj_tensor = torch.zeros(max_adjacent_vertices, device=self.device)
            len_adj = len(vertex.adj_vertices)
            _adj_tensor[:len_adj] = torch.tensor(list(vertex.adj_vertices), device=self.device)
            self.state[vertex_id, 1:max_adjacent_vertices+1] = _adj_tensor

        self.previous_state[:] = self.state[:, 0]
        self.max_adjacent_vertices = max_adjacent_vertices
        self.df = df

    def get_border_vertices(self):
        """
        Computes the vertices on the border of the graph. A vertex is on the border if it has an edge with a vertex
        from another district. vertex_on_border is a dict of tuples (vertex_id, district_id) -> tensors where vertex_id is the id of
        the vertex and district_id is the id of the district that the vertex is on the border with. This will be useful to know possible actions.
        """
        for vertex in self.vertices.values():
            for adj_vertex_id in vertex.adj_vertices:
                adj_vertex = self.vertices[adj_vertex_id]
                if vertex.district_id != adj_vertex.district_id:
                    self.update_border(vertex, adj_vertex)

    def change_vertex_district(self, prescinct_id, new_district_id, backward=False):
        """Changes the district of a vertex only if it is on a border with a vertex of new_district_id

        Args:
            vertex_id (int): The id of the vertex to change the district.
            new_district_id (int): The new district id.
        """
        _vertex = self.vertices[prescinct_id]
        old_district_id = _vertex.district_id
        if new_district_id == old_district_id:
            raise ValueError("The vertex is already in the district you want to change to")
        
        if (prescinct_id, new_district_id) not in self.vertex_on_border.keys():
            raise ValueError(f"The vertex is not on the border with the district you want to change to: {prescinct_id, new_district_id}")
        
        # we first change the district of the vertex and remove it from the vertex_on_border dict 
        # then we update the bordering_vertex with our new district dict
        self.vertices[prescinct_id].district_id = new_district_id
        self.state[prescinct_id, 0] = new_district_id

        list_to_check_possible_actions = [self.vertices[vert_id] for vert_id in self.vertices[prescinct_id].adj_vertices]
        list_to_check_possible_actions += [self.vertices[prescinct_id]]
        for vertex in list_to_check_possible_actions:
            # we remove all border information related to that vertex
            if self.district_id_on_border_of_vertex.get(vertex.prescinct_id) is not None:
                for adj_vertex_did in range(1, self.max_district_id+1):
                    # could be optimized but wtv
                    if self.vertex_on_border.get((vertex.prescinct_id, adj_vertex_did)) is not None:
                        self.vertex_on_border.pop((vertex.prescinct_id, adj_vertex_did))

                    district_on_border = self.district_id_on_border_of_vertex.get(vertex.prescinct_id)
                    if district_on_border is not None:
                        if len(district_on_border) > 0:
                            if adj_vertex_did in district_on_border:
                                district_on_border.remove(adj_vertex_did)

                    self.state[vertex.prescinct_id, self.max_adjacent_vertices+1:][adj_vertex_did-1] = 0
            self.state[vertex.prescinct_id, 0] = vertex.district_id

                    # self.state[vertex.prescinct_id, 1:][self.state[vertex.prescinct_id, 1:].abs() == adj_vertex_id] = adj_vertex_id

            
            # we recompute all border information related to changed vertex
            for adj_vertex_id in vertex.adj_vertices:
                adj_vertex = self.vertices[adj_vertex_id]
                if vertex.district_id != adj_vertex.district_id:
                    self.update_border(vertex, adj_vertex)

        if not backward:
            self.taken_actions[prescinct_id, new_district_id-1] = 1
        else:
            self.taken_actions[prescinct_id, old_district_id-1] = 0
        
        self.previous_state[prescinct_id] = old_district_id

    def update_border(self, vertex, adj_vertex):
        """Updates the border of the graph when a vertex changes its district.

        Args:
            vertex (Vertex): The vertex that changes its district.
            adj_vertex (Vertex): The adjacent vertex.
        """
        vertex_prescinct_id = vertex.prescinct_id
        # update possible actions
        self.vertex_on_border[(vertex_prescinct_id, adj_vertex.district_id)] = 1
                    
        # add state information about the bordering vertex by encoding with negative of id
        self.state[vertex_prescinct_id, 0] = -vertex.district_id
        self.state[vertex_prescinct_id, self.max_adjacent_vertices+1:][adj_vertex.district_id-1] = 1
        # self.state[vertex.prescinct_id, 1:][self.state[vertex.prescinct_id, 1:].abs() == adj_vertex.prescinct_id] = -adj_vertex.prescinct_id


        # update bordering district information
        if self.district_id_on_border_of_vertex.get(vertex_prescinct_id) is None:
            self.district_id_on_border_of_vertex[vertex_prescinct_id] = set()
        self.district_id_on_border_of_vertex[vertex_prescinct_id].add(adj_vertex.district_id)
    
    def get_full_state(self):
        state = self.state.clone().flatten()
        taken_actions = self.taken_actions.clone().flatten()

        return torch.cat((state, taken_actions), dim=0)
    
    def full_state_to_graph_state_taken_actions(self, full_state):
        state = full_state[:self.state.numel()].reshape(self.state.shape)
        taken_actions = full_state[self.state.numel():].reshape(self.taken_actions.shape)

        return state, taken_actions

    
            
class Vertex:
    def __init__(self, prescinct_id, district_id):
        """Initializes a vertex with a district_id according to which district the precinct belongs to.

        Args:
            prescinct_id (int): The precinct_id of the vertex.
            district_id (int): The district_id of the precinct.
        """
        self.prescinct_id = prescinct_id
        self.district_id = district_id
        self.adj_vertices = set()

    def __str__(self):
        return str(self.district_id)
    
    def set_adj(self, adj_list):
        """Sets an edge between this vertex and another vertex.

        Args:
            adj_list (List): The adjacent vertices to set the vertex
        """
        for vertex in adj_list:
            self.adj_vertices.add(vertex)