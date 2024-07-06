import pandas as pd
from neo4j import GraphDatabase

from src.shared import config

logger = config.get_logger("DatabaseWrapper")


class DatabaseWrapper:
    def __init__(self):

        logger.info("Connecting to the database")
        logger.debug(f"URI: {config.DB_URI}")
        logger.debug(f"User: {config.DB_USER}")
        self.driver = GraphDatabase.driver(config.DB_URI, auth=(config.DB_USER, config.DB_PASSWORD))

        # Create database indexes
        self.create_index("idIndex", "Publication", "id")
        self.create_vector_index("abstractEmbIndex", "Publication", "abstract_emb", 16)
        self.create_vector_index("titleEmbIndex", "Publication", "title_emb", 16)

    def create_vector_index(self, index, label, key, dimensions):
        logger.debug(
            f"Creating vector index '{index}' for label '{label}' on key '{key}' with '{dimensions}' dimensions.")
        with self.driver.session() as session:
            query = f"""
            CREATE VECTOR INDEX {index} IF NOT EXISTS
            FOR (n:{label})
            ON n.{key}
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

    def create_node(self, label, properties):
        with self.driver.session() as session:
            query = "CREATE (n:$label $props) RETURN n"
            session.run(query, label=label, props=properties)

    def merge_node(self, label: str, node_id: str, properties: dict):
        with self.driver.session() as session:
            query = f"""
            MERGE (n:{label} {{id: $id}})
            ON CREATE SET n += $properties
            ON MATCH SET n += $properties
            RETURN n
            """
            result = session.run(query, id=node_id, properties=properties)

            if result.single() is None:
                logger.error(f"Failed to create paper {node_id}")

    def merge_edge(self, node_label_1, node_label_2, edge_label: str, node_id_1: str, node_id_2: str, properties: dict):
        with self.driver.session() as session:
            query = f"""
            MATCH (n1:{node_label_1} {{id: $id1}})
            MATCH (n2:{node_label_2} {{id: $id2}})
            MERGE (n1)-[r:{edge_label}]-(n2)
            ON CREATE SET r += $properties
            ON MATCH SET r += $properties
            RETURN r
            """
            result = session.run(query, id1=node_id_1, id2=node_id_2, properties=properties)

            if result.single() is None:
                logger.error(f"Failed to create edge.")

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

    def get_similar_nodes_vec(self, label, key, vector, thresh, k) -> pd.DataFrame:
        with self.driver.session() as session:
            query = f"""
            MATCH (n:{label})
            WHERE gds.similarity.cosine(n.{key}, $vector) > {thresh}
            RETURN n.id AS id, gds.similarity.cosine(n.{key}, $vector) AS sim
            LIMIT $k
            """
            result = session.run(query, vector=vector, k=k).data()
            return pd.DataFrame(result)

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
