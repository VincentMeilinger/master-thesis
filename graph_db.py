from neo4j import GraphDatabase
import config


class GraphDB:

    def __init__(self):
        self.driver = GraphDatabase.driver(config.db_url, auth=(config.db_user, config.db_password))

    def close(self):
        self.driver.close()

    def create_nodes(self, label, properties_list):
        """
        Create nodes with a given label and list of property dictionaries.
        :param label: The label for the nodes.
        :param properties_list: A list of dictionaries, each representing the properties of a node.
        """
        with self.driver.session() as session:
            for properties in properties_list:
                session.write_transaction(self._create_node_tx, label, properties)

    @staticmethod
    def _create_node_tx(tx, label, properties):
        query = f"CREATE (n:{label} {{ {', '.join([f'{key}: ${key}' for key in properties.keys()])} }})"
        tx.run(query, **properties)

    def create_edges(self, from_label, to_label, relationship, pairs):
        """
        Create edges between nodes.
        :param from_label: Label of the source node.
        :param to_label: Label of the destination node.
        :param relationship: Type of relationship.
        :param pairs: List of tuples, where each tuple contains (source_node_properties, destination_node_properties).
        """
        with self.driver.session() as session:
            for from_props, to_props in pairs:
                session.write_transaction(self._create_edge_tx, from_label, to_label, relationship, from_props,
                                          to_props)

    @staticmethod
    def _create_edge_tx(tx, from_label, to_label, relationship, from_props, to_props):
        query = (
                f"MATCH (a:{from_label}), (b:{to_label}) "
                f"WHERE " + " AND ".join([f"a.{k} = ${'a_' + k}" for k in from_props.keys()]) + " AND " +
                " AND ".join([f"b.{k} = ${'b_' + k}" for k in to_props.keys()]) +
                f" CREATE (a)-[r:{relationship}]->(b)"
        )
        tx.run(query, **{f'a_{k}': v for k, v in from_props.items()}, **{f'b_{k}': v for k, v in to_props.items()})

    def delete_nodes(self, label, properties=None):
        """
        Delete nodes by label, optionally filtered by properties.
        :param label: The label of the nodes to delete.
        :param properties: Optional dictionary of properties to match for deletion.
        """
        with self.driver.session() as session:
            session.write_transaction(self._delete_nodes_tx, label, properties)

    @staticmethod
    def _delete_nodes_tx(tx, label, properties):
        properties_match = " AND ".join([f"n.{key} = ${key}" for key in properties.keys()]) if properties else ""
        query = f"MATCH (n:{label})" + (f" WHERE {properties_match}" if properties_match else "") + " DELETE n"
        tx.run(query, **properties)

    def update_nodes(self, label, match_properties, update_properties):
        """
        Update nodes matching certain properties with new properties.
        :param label: The label of the nodes to update.
        :param match_properties: The properties to match nodes on.
        :param update_properties: The new properties to set on matched nodes.
        """
        with self.driver.session() as session:
            session.write_transaction(self._update_nodes_tx, label, match_properties, update_properties)

    @staticmethod
    def _update_nodes_tx(tx, label, match_properties, update_properties):
        match_clause = " AND ".join([f"n.{key} = ${'match_' + key}" for key in match_properties.keys()])
        set_clause = ", ".join([f"n.{key} = ${'update_' + key}" for key in update_properties.keys()])
        query = f"MATCH (n:{label}) WHERE {match_clause} SET {set_clause}"
        tx.run(query, **{f'match_{k}': v for k, v in match_properties.items()},
               **{f'update_{k}': v for k, v in update_properties.items()})

