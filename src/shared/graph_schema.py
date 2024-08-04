from enum import Enum
import torch
import torch.nn.functional as F


class NodeType(Enum):
    """
    Enumeration of Knowledge Graph Nodes.
    """
    PUBLICATION = "Publication"
    AUTHOR = "Author"
    CO_AUTHOR = "CoAuthor"
    TRUE_AUTHOR = "TrueAuthor"
    ORGANIZATION = "Organization"
    VENUE = "Venue"

    def one_hot(self):
        return node_one_hot[self.value]


node_one_hot = {
    node_type.value: F.one_hot(torch.tensor(i), num_classes=len(NodeType)).type(torch.float32)
    for i, node_type in enumerate(NodeType)
}


class EdgeType(Enum):
    """
    Enumeration of Knowledge Graph Edges.
    """
    # n -[r:similar]-> m
    SIMILAR = "Similar"
    SIM_ORG = "SimilarOrg"
    SIM_VENUE = "SimilarVenue"
    SIM_TITLE = "SimilarTitle"
    SIM_ABSTRACT = "SimilarAbstract"
    SIM_KEYWORDS = "SimilarKeywords"
    SIM_YEAR = "SimilarYear"
    SIM_AUTHOR = "SimilarAuthor"

    # Publication -[r]-> n
    PUB_VENUE = "PubVenue"
    PUB_YEAR = "PubYear"
    PUB_AUTHOR = "PubAuthor"
    PUB_CO_AUTHOR = "PubCoAuthor"
    PUB_TRUE_AUTHOR = "PubTrueAuthor"
    PUB_CITES = "PubCites"
    PUB_ORG = "PubOrg"

    # Author -[r]-> n
    AUTHOR_ORG = "AuthorOrg"
    AUTHOR_PUB = "AuthorPub"
    AUTHOR_CO_AUTHOR = "AuthorCoAuthor"
    AUTHOR_TRUE_AUTHOR = "AuthorTrueAuthor"

    # Organization -[r]-> n
    ORG_PUB = "OrgPub"
    ORG_AUTHOR = "OrgAuthor"
    ORG_CO_AUTHOR = "OrgCoAuthor"
    ORG_TRUE_AUTHOR = "OrgTrueAuthor"

    # Venue -[r]-> n
    VENUE_PUB = "VenuePub"

    def one_hot(self):
        return edge_one_hot[self.value]


edge_one_hot = {
    edge_type.value: F.one_hot(torch.tensor(i), num_classes=len(EdgeType)).type(torch.float32)
    for i, edge_type in enumerate(EdgeType)
}
