import atexit
import pandas as pd
from neo4j import GraphDatabase
from typing import List, Dict, Any

from src.shared import config
from src.shared.graph_schema import NodeType, EdgeType

logger = config.get_logger("DatabaseWrapper")


class DatabaseWrapper:
    def __init__(self, database: str = None):
        logger.info("Connecting to the database ...")
        logger.debug(f"URI: {config.DB_URI}")
        logger.debug(f"User: {config.DB_USER}")
        try:
            self.driver = GraphDatabase.driver(config.DB_URI, auth=(config.DB_USER, config.DB_PASSWORD), database=database)
        except Exception as e:
            logger.error(f"Failed to connect to the database: {e}")
        # Create Node Indexes
        #for node_type in NodeType:
        #    self.create_index(f"index_{node_type.value}_id", node_type.value, "id")

        atexit.register(self.close)
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

    def create_nodes(self, node_type: NodeType, nodes: list):
        with self.driver.session() as session:
            query = f"""
            UNWIND $nodes AS node
            CREATE (n:{node_type.value} $properties)
            RETURN n
            """
            result = session.run(query, nodes=[{"id": node["id"], "properties": node} for node in nodes])

            if result.single() is None:
                logger.error(f"Failed to create nodes of type {type}")

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

    def merge_nodes(self, node_type: NodeType, nodes: list):
        with self.driver.session() as session:
            query = f"""
            UNWIND $nodes AS node
            MERGE (n:{node_type.value} {{id: node.id}})
            ON CREATE SET n += node.properties
            ON MATCH SET n += node.properties
            RETURN n
            """
            result = session.run(query, nodes=[{"id": node["id"], "properties": node} for node in nodes])

            if result.single() is None:
                logger.error(f"Failed to merge nodes of type {type}")

    def merge_edges(self, start_label: NodeType, end_label: NodeType, edge_type: EdgeType, edges: list):
        with self.driver.session() as session:
            query = f"""
            UNWIND $edges AS edge
            MATCH (a:{start_label.value} {{id: edge.start_id}})
            MATCH (b:{end_label.value} {{id: edge.end_id}})
            MERGE (a)-[r:{edge_type.value}]->(b)
            RETURN r
            """
            result = session.run(query, edges=[{
                "start_id": edge[0],
                "end_id": edge[1]
            } for edge in edges])

            if result.single() is None:
                logger.error(f"Failed to merge edges of type {edge_type.value}")

    def merge_edges_with_properties(self, start_label: NodeType, end_label: NodeType, edge_type: EdgeType, edges: list):
        with self.driver.session() as session:
            query = f"""
            UNWIND $edges AS edge
            MATCH (a:{start_label.value} {{id: edge.start_id}})
            MATCH (b:{end_label.value} {{id: edge.end_id}})
            MERGE (a)-[r:{edge_type.value}]->(b)
            ON CREATE SET r += edge.properties
            ON MATCH SET r += edge.properties
            RETURN r
            """
            result = session.run(query, edges=[{
                "start_id": edge[0],
                "end_id": edge[1],
                "properties": edge[2]
            } for edge in edges])

            if result.single() is None:
                logger.error(f"Failed to merge edges of type {edge_type.value}")

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
                return
                #logger.error(f"Failed to find and update node {node_id}")

    def merge_properties_batch(self, node_type: NodeType, nodes: List[Dict[str, Any]]):
        with self.driver.session() as session:
            query = f"""
            UNWIND $nodes AS node
            MATCH (n:{node_type.value} {{id: node.id}})
            SET n += node.properties
            RETURN n
            """
            result = session.run(query, nodes=nodes)

            if result.single() is None:
                logger.debug(f"Failed to find and update nodes")

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

    def iter_nodes(self, node_type: NodeType, attr_keys=None, batch_size: int=config.DB_BATCH_SIZE):
        if attr_keys is None:
            attr_keys = []

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

    def iter_nodes_with_edge_count(self, node_type: NodeType, count_edge_type: EdgeType, attr_keys=None, batch_size: int = config.DB_BATCH_SIZE):
        if attr_keys is None:
            attr_keys = []

        # Prepare the properties to return
        props = ", ".join([f"n.{key} AS {key}" for key in attr_keys])

        with self.driver.session() as session:
            offset = 0
            if props:
                query = f"""
                MATCH (n:{node_type.value})
                OPTIONAL MATCH (n)-[r:{count_edge_type.value}]-()
                RETURN {props}, COUNT(r) AS edge_count
                SKIP $offset LIMIT $batch_size
                """
            else:
                query = f"""
                MATCH (n:{node_type.value})
                OPTIONAL MATCH (n)-[r:{count_edge_type.value}]-()
                RETURN n, COUNT(r) AS edge_count
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

    def get_nodes_by_id(self, label: str, ids: list, keys: list) -> pd.DataFrame:
        with self.driver.session() as session:
            props = ", ".join([f"n.{key} AS {key}" for key in keys])
            query = f"""
            MATCH (n:{label})
            WHERE n.id IN $ids
            RETURN {props}
            """
            result = session.run(query, ids=ids).data()
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
                  relationshipFilter: '<'
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
                query = """
                CALL apoc.periodic.iterate(
                "MATCH (n) RETURN n",
                "DETACH DELETE n",
                {batchSize:10000})
                """
                session.run(query)
                logger.info("Deleted all nodes.")
        except Exception as e:
            logger.error(f"Failed to delete all nodes: {e}")

    def delete_nodes(self, label):
        with self.driver.session() as session:
            query = f"""
            CALL apoc.periodic.iterate(
            "MATCH (n:{label}) RETURN n",
            "DETACH DELETE n",
            {{batchSize:10000}})
            """
            session.run(query)

    def delete_edges(self, edge_type: EdgeType):
        with self.driver.session() as session:
            query = f"""
            CALL apoc.periodic.iterate(
            "MATCH ()-[r:{edge_type.value}]-() RETURN r",
            "DELETE r",
            {{batchSize:10000}})
            """
            session.run(query)

    def delete_edges_for_empty_attr(self, node_type: NodeType, edge_type: EdgeType, attr: str):
        with self.driver.session() as session:
            query = f"""
            MATCH (n:{node_type.value})-[r:{edge_type.value}]->()
            WHERE n.{attr} = ''
            DELETE r
            """
            session.run(query)

    def count_nodes(self, node_type):
        with self.driver.session() as session:
            query = f"""
            MATCH (n:{node_type.value})
            RETURN count(n) as count
            """
            result = session.run(query).single()
            return result["count"]

    def count_edges(self, node_type, node_id, edge_type):
        with self.driver.session() as session:
            query = f"""
            MATCH (n:{node_type.value})-[r:{edge_type.value}]-()
            WHERE n.id = $id
            RETURN count(r) as count
            """
            result = session.run(query, id=node_id)
            count = result.single()["count"]
            return count

    def close(self):
        logger.info("Closing the database connection")
        self.driver.close()

    def custom_query(self, query, parameters=None):
        logger.debug(f"Running custom query: {query} with parameters: {parameters}")
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return result.data()