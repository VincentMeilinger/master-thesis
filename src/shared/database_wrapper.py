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
        logger.info(f"Creating vector index '{index}' for label '{label}' on key '{key}' with '{dimensions}' dimensions.")
        with self.driver.session() as session:
            # Define the Cypher query
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
        logger.info(f"Creating index '{index}' for label '{label}' on key '{key}'.")
        with self.driver.session() as session:
            # Define the Cypher query
            query = f"""
            CREATE INDEX {index} IF NOT EXISTS
            FOR (n:{label})
            ON (n.{key})
            """
            session.run(query)

    def close(self):
        logger.info("Closing the database connection")
        self.driver.close()

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

    def iterate_all_papers(self, batch_size):
        with self.driver.session() as session:
            offset = 0
            while True:
                result = session.run("""
                MATCH (n)
                RETURN n
                SKIP $offset LIMIT $batch_size
                """, offset=offset, batch_size=batch_size)

                nodes = [record["n"] for record in result]

                if not nodes:
                    break

                yield nodes
                offset += batch_size

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

