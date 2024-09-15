import torch
from graphdatascience import GraphDataScience
from neo4j import GraphDatabase, Result
from torch_geometric.data import HeteroData

from src.shared.graph_schema import NodeType, EdgeType, node_one_hot, edge_one_hot, edge_val_to_pyg_key_vals
from src.shared import config

class GraphSampling:
    def __init__(self, node_spec: list, edge_spec: list, node_properties: list):
        self.driver = GraphDatabase.driver(config.DB_URI, auth=(config.DB_USER, config.DB_PASSWORD))
        self.node_spec = node_spec
        self.edge_spec = edge_spec
        self.node_properties = node_properties

    def random_nodes(self, node_type: NodeType, node_properties: list, n: int):
        with self.driver.session() as session:
            prop_str = ', '.join([f'n.{prop} as {prop}' for prop in node_properties])
            query = f"""
                MATCH (n:{node_type.value})
                RETURN n.id as id, {prop_str}
                LIMIT {n}
            """
            result = session.run(query)
            data = result.data()
            return data

    def n_hop_neighbourhood(
            self,
            start_node_type: NodeType,
            start_node_id: str,
            node_types: list = None,
            edge_types: list = None,
            max_level: int = 6
    ):
        with self.driver.session() as session:
            node_filter = '|'.join(
                [nt.value for nt in self.node_spec] if node_types is None else
                [nt.value for nt in node_types]
            )
            edge_filter = '|'.join(
                [f"<{et.value}" for et in self.edge_spec] if edge_types is None else
                [f"<{et.value}" for et in edge_types]
            )

            query = f"""
                    MATCH (start:{start_node_type.value} {{id: '{start_node_id}'}})
                    CALL apoc.path.subgraphAll(start, {{
                      maxLevel: {max_level},
                      relationshipFilter: '{edge_filter}',
                      labelFilter: '+{node_filter}'
                    }}) YIELD nodes, relationships
                    RETURN nodes, relationships
                """
            result = session.run(query)
            data = result.single()
            return data


    @staticmethod
    def neo_to_pyg(
        data,
        node_attr: str
    ):
        if not data:
            return None, None

        nodes = data["nodes"]
        relationships = data["relationships"]

        print(f"Nodes: {len(nodes)}, Relationships: {len(relationships)}")
        if len(nodes) > 500:
            print(f"Too many nodes: {len(nodes)}")
            return None, None

        # Create data object
        h_data = HeteroData()

        node_features = {}
        node_ids = {}
        node_id_map = {}

        for node in nodes:
            node_id = node.get("id")
            node_feature = node.get(node_attr, None)
            if node_feature is None:
                print(f"Node {node_id} has no attribute {node_attr}")
                continue
            node_label = list(node.labels)[0]
            if node_label not in node_features:
                node_features[node_label] = []
                node_ids[node_label] = []

            # Convert node features to tensors
            node_features[node_label].append(torch.tensor(node_feature, dtype=torch.float32))
            node_ids[node_label].append(node_id)

            # Map node id to its index in the list
            node_id_map[node_id] = len(node_ids[node_label]) - 1

        # Convert list of features to a single tensor per node type
        for node_label, node_features in node_features.items():
            h_data[node_label].x = torch.vstack(node_features)

        # Process relationships
        edge_dict = {}

        for rel in relationships:
            key = edge_val_to_pyg_key_vals[rel.type]
            if key not in edge_dict:
                edge_dict[key] = [[], []]

            source_id = rel.start_node.get("id")
            target_id = rel.end_node.get("id")

            # Append the indices of the source and target nodes
            edge_dict[key][0].append(node_id_map[source_id])
            edge_dict[key][1].append(node_id_map[target_id])

        # Convert edge lists to tensors
        for key in edge_dict:
            h_data[key[0], key[1], key[2]].edge_index = torch.vstack([
                torch.tensor(edge_dict[key][0], dtype=torch.long),
                torch.tensor(edge_dict[key][1], dtype=torch.long)
            ])

            h_data[key[0], key[1], key[2]].edge_attr = torch.vstack(
                [edge_one_hot[key[1]] for _ in range(len(edge_dict[key][0]))])

        # If the total number of edges is smaller than 10, return None
        if sum([len(edge_dict[key][0]) for key in edge_dict]) < 10:
            return None, None

        return h_data, node_id_map