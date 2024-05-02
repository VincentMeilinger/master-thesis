from src import DatabaseWrapper
from src import config

logger = config.get_logger("PopulateDB")


def populate_who_is_who(data):
    """Expects a list of dictionaries in the form of the WhoIsWho dataset."""
    # Create a new graph database
    db = DatabaseWrapper()

    logger.debug("Creating nodes in neo4j graph database ...")
    for paper_id, paper_info in data.items():
        # Add the paper node with or without attributes
        authors = paper_info.pop('authors')
        db.merge_paper(paper_info)
        for author_info in authors:
            db.merge_author(author_info)
            db.merge_author_paper_relationship(
                author_info['name'],
                author_info['org'],
                paper_info['id']
            )
    logger.debug("Done.")