import re
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from src.shared import config
from src.shared.database_wrapper import DatabaseWrapper
from src.shared.graph_schema import NodeType, EdgeType


logger = config.get_logger("LinkNodes")


def get_sample_size(num_papers):
    random_k = 1 + random.random() * 0.2
    return int(num_papers ** 0.5 * random_k)


def link_true_authors(db: DatabaseWrapper, train_data, config_file):
    max_iterations = config_file.get("max_iterations", None)
    node_type = NodeType.PUBLICATION
    edges_to_merge = []
    current_iteration = 0

    with tqdm(total=max_iterations, desc="Merging true-author edges") as pbar:
        for author_id, values in train_data.items():
            papers = values.get('normal_data', [])
            print(f"Number of papers: {len(papers)}")

            pbar.update(len(papers) + len(values.get('outliers', [])))
            if max_iterations is not None and current_iteration >= max_iterations:
                break
            current_iteration += 1

            for i in range(len(papers)):
                for j in range(len(papers)):
                    if i == j:
                        continue
                    edges_to_merge.append([papers[i], papers[j]])

            print(f"Number of edges to merge: {len(edges_to_merge)}")
            # Randomly sample 10-40% of the edges_to_merge list
            sample_size = get_sample_size(len(edges_to_merge))
            print(f"Sample size: {sample_size}")
            edges_to_merge = random.sample(edges_to_merge, sample_size)
            print(f"Sampled number of edges to merge: {len(edges_to_merge)}")

            if len(edges_to_merge) > 1000:
                db.merge_edges(start_label=node_type, end_label=node_type, edge_type=EdgeType.SAME_AUTHOR,
                               edges=edges_to_merge)
                edges_to_merge.clear()

        if edges_to_merge:
            db.merge_edges(start_label=node_type, end_label=node_type, edge_type=EdgeType.SAME_AUTHOR,
                           edges=edges_to_merge)
            edges_to_merge.clear()


def link_co_author_network(db: DatabaseWrapper, data: dict, config: dict):
    node_type = NodeType.PUBLICATION

    co_author_overlap_threshold = config["link_node"]["co_author_overlap_threshold"]
    num_nodes = db.count_nodes(node_type)
    attrs = ['id']
    co_author_map = defaultdict(list)
    co_author_overlap = defaultdict(int)

    print(f"Linking {node_type.value} nodes based on co-authorship ...")
    with tqdm(total=num_nodes, desc=f"Progress {node_type.value} co-authorship") as pbar:
        for nodes in db.iter_nodes(node_type, attrs):
            for node in nodes:
                co_authors = [author["name"] for author in data[node['id']]['authors']]
                for author in co_authors:
                    name = author.strip()
                    name = re.sub(r'[^A-Za-z\s]', '', name)
                    name_parts = name.split()
                    if len(name_parts) == 0:
                        continue

                    surname = name_parts[-1]
                    given_name_initial = (name_parts[0] if len(name_parts) > 1 else ' ')[0]
                    abbrev = f"{surname} {given_name_initial}"
                    co_author_map[abbrev].append(node['id'])
                pbar.update(1)

    for k, v in co_author_map.items():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                co_author_overlap[(v[i], v[j])] += 1

    for k, v in co_author_overlap.items():
        total_num_authors = data[k[0]]["authors"] + data[k[1]]["authors"]
        co_author_overlap[k] = v / len(total_num_authors)

    print(f"Max. co-authors: {max(len(v) for v in co_author_map.values())}")
    print(f"Max. co-author overlap: {max(co_author_overlap.values())}")

    print("Number of co-author pairs:", len(co_author_overlap))
    print(f"Number of co-author pairs with overlap > {co_author_overlap_threshold}:",
          len([v for v in co_author_overlap.values() if v > co_author_overlap_threshold]))

    print("Merging edges ...")
    edges_to_merge = [[k[0], k[1], {'sim': v}] for k, v in co_author_overlap.items() if v > co_author_overlap_threshold]
    with tqdm(total=len(edges_to_merge), desc="Merging co-author edges") as pbar:
        for i in range(0, len(edges_to_merge), 1000):
            db.merge_edges_with_properties(start_label=node_type, end_label=node_type, edge_type=EdgeType.SIM_AUTHOR,
                                           edges=edges_to_merge[i:i + 1000])
            pbar.update(1000)


def link_node_attr_cosine(db: DatabaseWrapper, node_type: NodeType, vec_attr: str, edge_type: EdgeType,
                          threshold: float = 0.7, filter_empty_original_attr: str = None, k: int = 8):
    num_nodes = db.count_nodes(node_type)
    edges = []
    attrs = ['id', vec_attr]
    if filter_empty_original_attr:
        attrs.append(filter_empty_original_attr)

    print(f"Linking {node_type.value} nodes based on {vec_attr} attribute ...")
    with tqdm(total=num_nodes, desc=f"Progress {node_type.value} {vec_attr}") as pbar:
        for nodes in db.iter_nodes(node_type, attrs):
            for node in nodes:
                if filter_empty_original_attr and not node[filter_empty_original_attr]:
                    pbar.update(1)
                    print(f"Skipping node {node['id']} because {filter_empty_original_attr} is empty")
                    continue

                similar_nodes = db.get_similar_nodes_vec(
                    node_type,
                    vec_attr,
                    node[vec_attr],
                    threshold,
                    k
                )
                for ix, row in similar_nodes.iterrows():
                    if row['id'] == node['id']:
                        continue
                    edges.append([node['id'], row['id']])
                    # db.merge_edge(node_type, node['id'], node_type, row['id'], edge_type, {"sim": row['sim']})
                if len(edges) > 1000:
                    print(f"Merging {len(edges)} edges ...")
                    db.merge_edges(start_label=node_type, end_label=node_type, edge_type=edge_type, edges=edges)
                    edges.clear()

                pbar.update(1)
    if edges:
        db.merge_edges(start_label=node_type, end_label=node_type, edge_type=edge_type, edges=edges)

def link_all_attributes(db: DatabaseWrapper, model, config: dict):
    model_dim = model.get_sentence_embedding_dimension()

    logger.info("Creating links based on attribute similarities ...")
    db.create_vector_index('title_index', NodeType.PUBLICATION, 'title_emb', model_dim)
    link_node_attr_cosine(
        db,
        NodeType.PUBLICATION,
        'title_emb',
        EdgeType.SIM_TITLE,
        threshold=config["link_node"]["link_title_threshold"],
        k=config["link_node"]["link_title_k"]
    )

    db.create_vector_index('abstract_index', NodeType.PUBLICATION, 'abstract_emb', model_dim)
    link_node_attr_cosine(
        db,
        NodeType.PUBLICATION,
        'abstract_emb',
        EdgeType.SIM_ABSTRACT,
        threshold=config["link_node"]["link_abstract_threshold"],
        k=config["link_node"]["link_abstract_k"]
    )

    db.create_vector_index('venue_index', NodeType.PUBLICATION, 'venue_emb', model_dim)
    link_node_attr_cosine(
        db,
        NodeType.PUBLICATION,
        'venue_emb',
        EdgeType.SIM_VENUE,
        threshold=config["link_node"]["link_venue_threshold"],
        k=config["link_node"]["link_venue_k"]
    )

    db.create_vector_index('org_index', NodeType.PUBLICATION, 'org_emb', model_dim)
    link_node_attr_cosine(
        db,
        NodeType.PUBLICATION,
        'org_emb',
        EdgeType.SIM_ORG,
        threshold=config["link_node"]["link_org_threshold"],
        k=config["link_node"]["link_org_k"]
    )

    # Delete edges if the publication attribute is empty to avoid false links.
    logger.info("Deleting links for empty attributes ...")
    db.delete_edges_for_empty_attr(NodeType.PUBLICATION, EdgeType.SIM_TITLE, 'title')

    db.delete_edges_for_empty_attr(NodeType.PUBLICATION, EdgeType.SIM_ABSTRACT, 'abstract')

    db.delete_edges_for_empty_attr(NodeType.PUBLICATION, EdgeType.SIM_VENUE, 'venue')

    db.delete_edges_for_empty_attr(NodeType.PUBLICATION, EdgeType.SIM_ORG, 'org')

    logger.info("Finished linking nodes.")