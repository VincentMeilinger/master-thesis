import pandas as pd
from neo4j import GraphDatabase
from typing import List, Dict, Any

from src.shared import config
from src.shared.graph_schema import NodeType, EdgeType

logger = config.get_logger("DatabaseWrapper")


class DatabaseWrapper:
    def __init__(self):
        logger.info("Connecting to the database ...")
        logger.debug(f"URI: {config.DB_URI}")
        logger.debug(f"User: {config.DB_USER}")
        self.driver = GraphDatabase.driver(config.DB_URI, auth=(config.DB_USER, config.DB_PASSWORD))

        logger.info("Database ready.")

    def create_vector_index(self, index_name: str, node_type: NodeType, attr_key: str, dimensions):
        logger.debug(
            f"Creating vector index '{index_name}' for label '{node_type.value}' on key '{attr_key}' with '{dimensions}' dimensions.")
        with self.driver.session() as session:
            query = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{node_type.value})
            ON n.{attr_key}
            OPTIONS {{indexConfig: {{
             `vector.dimensions`: {dimensions},
             `vector.similarity_function`: 'cosine'
            }}}}
            """
            session.run(query)

    def create_index(self, index, label, key):
        logger.debug(f"Creating index '{index}' for label '{label}' on key '{key}'.")
        with self.driver.session() as session:
            query = f"""
            CREATE INDEX {index} IF NOT EXISTS
            FOR (n:{label})
            ON (n.{key})
            """
            session.run(query)

    def create_node(self, type: NodeType, properties):
        with self.driver.session() as session:
            query = "CREATE (n:$label $props) RETURN n"
            session.run(query, label=type.value, props=properties)

    def merge_node(self, type: NodeType, node_id: str, properties: dict = {}):
        with self.driver.session() as session:
            query = f"""
            MERGE (n:{type.value} {{id: $id}})
            ON CREATE SET n += $properties
            ON MATCH SET n += $properties
            RETURN n
            """
            result = session.run(query, id=node_id, properties=properties)

            if result.single() is None:
                logger.error(f"Failed to create paper {node_id}")

    def merge_nodes(self, type: NodeType, nodes: List[Dict[str, Any]]):
        assert "id" in nodes[0] and "properties" in nodes[0], "Nodes should be a list of dictionaries with 'id' and 'properties' keys"
        with self.driver.session() as session:
            query = f"""
            UNWIND $nodes AS node
            MERGE (n:{type.value} {{id: node.id}})
            ON CREATE SET n += node.properties
            ON MATCH SET n += node.properties
            RETURN n
            """
            result = session.run(query, nodes=nodes)

            # Handling the results or logging errors
            if result.single() is None:
                logger.error("Failed to merge nodes")
            else:
                logger.info("Nodes merged successfully")

    def merge_edge(self, node_type_1: NodeType, node_id_1: str, node_type_2: NodeType, node_id_2: str, edge_type: EdgeType, properties: dict = {}):
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH (n1:{node_type_1.value} {{id: $id1}})
                MATCH (n2:{node_type_2.value} {{id: $id2}})
                MERGE (n1)-[r:{edge_type.value}]-(n2)
                ON CREATE SET r += $properties
                ON MATCH SET r += $properties
                RETURN r
                """
                result = session.run(query, id1=node_id_1, id2=node_id_2, properties=properties)

                if result.single() is None:
                    logger.error(f"Failed to create edge between {node_id_1} and {node_id_2}")
        except Exception as e:
            logger.exception(f"Failed to create edge between {node_id_1} and {node_id_2}")
            logger.exception(e)

    def merge_properties(self, type: NodeType, node_id: str, properties: dict = {}):
        with self.driver.session() as session:
            query = f"""
            MATCH (n:{type.value} {{id: $id}})
            SET n += $properties
            RETURN n
            """
            result = session.run(query, id=node_id, properties=properties)

            if result.single() is None:
                logger.error(f"Failed to find and update node {node_id}")

    def iterate_all_papers(self, batch_size: int):
        with self.driver.session() as session:
            offset = 0
            while True:
                query = """
                MATCH (n)
                RETURN n
                SKIP $offset LIMIT $batch_size
                """
                result = session.run(query, offset=offset, batch_size=batch_size)

                nodes = [record["n"] for record in result]

                if not nodes:
                    break

                yield nodes
                offset += batch_size

    def iter_nodes(self, node_type: NodeType, attr_keys: List[str] = [], batch_size: int=config.DB_BATCH_SIZE):
        props = ", ".join([f"n.{key} AS {key}" for key in attr_keys])
        with self.driver.session() as session:
            offset = 0
            if props:
                query = f"""
                MATCH (n:{node_type.value})
                RETURN {props}
                SKIP $offset LIMIT $batch_size
                """
            else:
                query = f"""
                MATCH (n:{node_type.value})
                RETURN n
                SKIP $offset LIMIT $batch_size
                """
            while True:
                result = session.run(query, offset=offset, batch_size=batch_size).data()

                if not result:
                    break

                yield result
                offset += batch_size

    def iterate_all_papers_get_properties(self, label: str, keys: list, batch_size: int):
        props = ", ".join([f"n.{key} AS {key}" for key in keys])
        with self.driver.session() as session:
            offset = 0
            while True:
                query = f"""
                MATCH (n:{label})
                RETURN {props}
                SKIP $offset LIMIT $batch_size
                """

                result = session.run(query, offset=offset, batch_size=batch_size).data()

                # Check if there are no more nodes
                if not result:
                    break

                yield pd.DataFrame(result)
                offset += batch_size

    def get_all_nodes_and_properties(self, label: str, keys: list) -> pd.DataFrame:
        with self.driver.session() as session:
            props = ", ".join([f"n.{key} AS {key}" for key in keys])
            query = f"""
            MATCH (n:{label})
            RETURN {props}
            """
            result = session.run(query).data()
            return pd.DataFrame(result)

    def get_similar_nodes_vec(self, node_type: NodeType, vec_attr: str, vector, thresh, k) -> pd.DataFrame:
        with self.driver.session() as session:
            query = f"""
            MATCH (n:{node_type.value})
            WHERE gds.similarity.cosine(n.{vec_attr}, $vector) > {thresh}
            RETURN n.id AS id, gds.similarity.cosine(n.{vec_attr}, $vector) AS sim
            LIMIT $k
            """
            result = session.run(query, vector=vector, k=k).data()
            return pd.DataFrame(result)

    def fetch_neighborhood(self, start_node_type: NodeType, start_node_id: str, max_level: int):
        with self.driver.session() as session:
            query = f"""
                    MATCH (start:{start_node_type.value} {{id: '{start_node_id}'}})
                    CALL apoc.path.subgraphAll(start, {{
                      maxLevel: {max_level},
                      relationshipFilter: '<>'
                    }}) YIELD nodes, relationships
                    RETURN nodes, relationships
                """
            result = session.run(query)

            nodes_list = []
            relationships_list = []

            for record in result:
                nodes_list.extend(record["nodes"])
                relationships_list.extend(record["relationships"])

        return nodes_list, relationships_list

    def delete_all_nodes(self):
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Deleted all nodes.")
        except Exception as e:
            logger.error(f"Failed to delete all nodes: {e}")

    def delete_nodes(self, label):
        with self.driver.session() as session:
            session.run(f"MATCH (n:{label}) DETACH DELETE n")

    def close(self):
        logger.info("Closing the database connection")
        self.driver.close()

    def custom_query(self, query, parameters=None):
        logger.debug(f"Running custom query: {query} with parameters: {parameters}")
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return result.data()
