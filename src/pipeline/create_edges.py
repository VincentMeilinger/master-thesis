from src.shared import config
from src.shared.run_config import RunConfig
from src.shared.pipeline_state import PipelineState
from src.shared.database_wrapper import DatabaseWrapper

logger = config.get_logger("CreateEdges")


def create_emb_sim_edges(db: DatabaseWrapper, run_config: RunConfig):
    for batch in db.iterate_all_papers_get_properties(
            "Publication",
            ["id", "abstract", "abstract_emb", "title_emb"],
            run_config.create_edges.batch_size
    ):
        logger.info(f"Processing batch of {len(batch)} nodes ...")
        for ix1, row1 in batch.iterrows():
            # If abstract length is less than 6 characters, skip
            if len(row1["abstract"]) < 6:
                continue

            # Similarity based on abstract embeddings
            sim_nodes_abstract = db.get_similar_nodes_vec(
                "Publication",
                "abstract_emb",
                row1["abstract_emb"],
                run_config.create_edges.similarity_threshold,
                run_config.create_edges.k_nearest_limit
            )

            for ix2, row2 in sim_nodes_abstract.iterrows():
                if row1["id"] == row2["id"]:  # Skip self
                    continue

                # Merge edge
                db.merge_edge(
                    "Publication",
                    "Publication",
                    "sim_abstract",
                    row1["id"],
                    row2["id"],
                    {"sim": row2["sim"]}
                )


def create_edges():
    """Create edges between nodes in the graph database based on author, venue, and keyword similarity.
    """
    run_config = RunConfig(config.RUN_DIR)
    db = DatabaseWrapper()
    state = PipelineState(config.RUN_ID, config.RUN_DIR)
    if not state.create_edges.state == 'completed':
        create_emb_sim_edges(db, run_config)
        state.create_edges.state = 'completed'
        state.save()



