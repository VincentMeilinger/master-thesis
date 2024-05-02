from neo4j import GraphDatabase
from src.shared import config


class DatabaseWrapper:

    def __init__(self):
        self.logger = config.get_logger("DatabaseWrapper")
        self.logger.info("Connecting to the database")
        self.logger.debug(f"URI: {config.db_uri}")
        self.logger.debug(f"User: {config.db_user}")
        self.driver = GraphDatabase.driver(config.db_uri, auth=(config.db_user, config.db_password))

    def close(self):
        self.logger.info("Closing the database connection")
        self.driver.close()

    def create_node_with_dict(self, label, properties):
        with self.driver.session() as session:
            # Define the Cypher query
            cypher_query = f"CREATE (n:{label} $props) RETURN n"

            # Execute the query
            session.run(cypher_query, props=properties)

    def merge_paper(self, properties):
        with self.driver.session() as session:
            # Construct the query dynamically based on the label and properties
            cypher_query = """
            MERGE (n:Paper {id: $id})
            ON CREATE SET n += $properties
            ON MATCH SET n += $properties
            RETURN n
            """

            result = session.run(cypher_query, id=properties['id'], properties=properties)

            if result.single() is None:
                self.logger.error(f"Failed to create paper {properties['id']}")

    def merge_author(self, properties):
        with self.driver.session() as session:
            # Construct the query dynamically based on the label and properties
            cypher_query = """
            MERGE (n:Author {name: $name, org: $org})
            ON CREATE SET n += $properties
            ON MATCH SET n += $properties
            RETURN n
            """

            result = session.run(cypher_query, name=properties['name'], org=properties['org'], properties=properties)

            if result.single() is None:
                self.logger.error(f"Failed to create author {properties['name']}")

    def merge_author_paper_relationship(self, author_name, author_org, paper_id):
        with self.driver.session() as session:
            cypher_query = """
            MATCH (author:Author {name: $name, org: $org}), (paper:Paper {id: $paperId})
            MERGE (author)-[r:wrote]->(paper)
            RETURN r
            """
            result = session.run(cypher_query, name=author_name, org=author_org, paperId=paper_id)

            if result.single() is None:
                self.logger.error(f"Failed to create relationship between {author_name} and {paper_id}")

    def create_nodes(self, label, properties_list):
        self.logger.debug(f"Creating nodes with label {label}")
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
        self.logger.debug(f"Creating edges from {from_label} to {to_label} with relationship {relationship}")
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
        self.logger.debug(f"Deleting nodes with label {label}")
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
        self.logger.debug(f"Updating nodes with label {label}")
        with self.driver.session() as session:
            session.write_transaction(self._update_nodes_tx, label, match_properties, update_properties)

    @staticmethod
    def _update_nodes_tx(tx, label, match_properties, update_properties):
        match_clause = " AND ".join([f"n.{key} = ${'match_' + key}" for key in match_properties.keys()])
        set_clause = ", ".join([f"n.{key} = ${'update_' + key}" for key in update_properties.keys()])
        query = f"MATCH (n:{label}) WHERE {match_clause} SET {set_clause}"
        tx.run(query, **{f'match_{k}': v for k, v in match_properties.items()},
               **{f'update_{k}': v for k, v in update_properties.items()})

