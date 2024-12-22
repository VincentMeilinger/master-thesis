import os
import json
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from ..shared import config
from ..datasets.who_is_who import WhoIsWhoDataset
from ..shared.graph_schema import NodeType
from ..shared.database_wrapper import DatabaseWrapper

logger = config.get_logger('CreateKG')


def process_batch(db, model, batch):
    if not batch[NodeType.PUBLICATION]:
        return
    title_embs = model.encode(
        [node['title'] for node in batch[NodeType.PUBLICATION]]
    )
    abstract_embs = model.encode(
        [node['abstract'] for node in batch[NodeType.PUBLICATION]]
    )
    venue_embs = model.encode(
        [node['venue'] for node in batch[NodeType.PUBLICATION]]
    )
    org_embs = model.encode(
        [node['org'] for node in batch[NodeType.PUBLICATION]]
    )
    for i, node in enumerate(batch[NodeType.PUBLICATION]):
        node['title_emb'] = title_embs[i]
        node['abstract_emb'] = abstract_embs[i]
        node['venue_emb'] = venue_embs[i]
        node['org_emb'] = org_embs[i]
        # vertically stack the embeddings
        node['feature_vec'] = list(title_embs[i]) + list(abstract_embs[i])
    db.merge_nodes(NodeType.PUBLICATION, batch[NodeType.PUBLICATION])
    batch[NodeType.PUBLICATION] = []


def create_nodes(db: DatabaseWrapper, model, data: dict, train_data: dict, config: dict):
    max_iterations = config.get('max_iterations', None)

    batch_nodes = defaultdict(list)
    current_iteration = 0
    authors_in_graph = set()

    with tqdm(total=max_iterations) as pbar:
        for author_id, values in train_data.items():
            if max_iterations is not None and current_iteration >= max_iterations:
                break

            authors_in_graph.add(author_id)

            papers = values.get('normal_data', [])
            papers.extend(values.get('outliers', []))

            current_iteration += 1

            for paper_id in papers:
                values = data[paper_id]
                authors = values.get('authors', [])
                org = ''
                if len(authors) > 0 and 'org' in authors[0]:
                    org = authors[0].get('org', '')
                paper_node = {
                    'id': values['id'],
                    'title': values['title'],
                    'abstract': values['abstract'],
                    'year': values['year'],
                    'venue': values['venue'],
                    'org': org
                }
                batch_nodes[NodeType.PUBLICATION].append(paper_node)

                if len(batch_nodes[NodeType.PUBLICATION]) % 1000 == 0:
                    process_batch(db, model, batch_nodes)

            pbar.update(1)

    process_batch(db, model, batch_nodes)
    print("Number of authors in the graph:", len(authors_in_graph))
    print("Number of publication nodes:", db.count_nodes(NodeType.PUBLICATION))


def reverse_dict(author_dict):
    paper_to_author = {}
    for author_id, values in author_dict.items():
        normal_papers = values.get('normal_data', [])
        for paper_id in normal_papers:
            paper_to_author[paper_id] = author_id
    return paper_to_author


def add_true_authors(db: DatabaseWrapper, train_data):
    paper_id_to_author = reverse_dict(train_data)

    with tqdm(total=db.count_nodes(NodeType.PUBLICATION), desc="Merging WhoIsWho train_author.json") as pbar:
        for nodes in db.iter_nodes(NodeType.PUBLICATION, ['id']):
            for node in nodes:
                true_author_id = paper_id_to_author.get(node['id'], '')
                true_author_name = train_data.get(true_author_id, {}).get('name', '')

                db.merge_properties(
                    type=NodeType.PUBLICATION,
                    node_id=node['id'],
                    properties={'true_author_id': true_author_id, 'true_author_name': true_author_name}
                )
                pbar.update(1)
