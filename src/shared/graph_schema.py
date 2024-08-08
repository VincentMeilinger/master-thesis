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
    SIM_PUB = "SimilarPub"
    SIM_ORG = "SimilarOrg"
    SIM_VENUE = "SimilarVenue"
    SIM_TITLE = "SimilarTitle"
    SIM_ABSTRACT = "SimilarAbstract"
    SIM_KEYWORDS = "SimilarKeywords"
    SIM_YEAR = "SimilarYear"
    SIM_AUTHOR = "SimilarAuthor"

    # Publication -[r]-> n
    PUB_VENUE = "PubVenue"
    PUB_AUTHOR = "PubAuthor"
    PUB_CO_AUTHOR = "PubCoAuthor"
    PUB_TRUE_AUTHOR = "PubTrueAuthor"
    PUB_CITES = "PubCites"
    PUB_ORG = "PubOrg"

    # Author -[r]-> n
    AUTHOR_ORG = "AuthorOrg"
    AUTHOR_PUB = "AuthorPub"
    AUTHOR_CO_AUTHOR = "AuthorCoAuthor"

    # CoAuthor -[r]-> n
    CO_AUTHOR_PUB = "CoAuthorPub"
    CO_AUTHOR_AUTHOR = "CoAuthorAuthor"
    CO_AUTHOR_ORG = "CoAuthorOrg"

    # Organization -[r]-> n
    ORG_PUB = "OrgPub"
    ORG_AUTHOR = "OrgAuthor"
    ORG_CO_AUTHOR = "OrgCoAuthor"

    # Venue -[r]-> n
    VENUE_PUB = "VenuePub"

    def one_hot(self):
        return edge_one_hot[self.value]

    def start_end(self):
        return edge_start_end[self]


edge_start_end = {
    # Similar edges
    EdgeType.SIM_PUB: (NodeType.PUBLICATION, NodeType.PUBLICATION),
    EdgeType.SIM_ORG: (NodeType.ORGANIZATION, NodeType.ORGANIZATION),
    EdgeType.SIM_VENUE: (NodeType.VENUE, NodeType.VENUE),
    EdgeType.SIM_TITLE: (NodeType.PUBLICATION, NodeType.PUBLICATION),
    EdgeType.SIM_ABSTRACT: (NodeType.PUBLICATION, NodeType.PUBLICATION),
    EdgeType.SIM_KEYWORDS: (NodeType.PUBLICATION, NodeType.PUBLICATION),
    EdgeType.SIM_YEAR: (NodeType.PUBLICATION, NodeType.PUBLICATION),
    EdgeType.SIM_AUTHOR: (NodeType.AUTHOR, NodeType.AUTHOR),

    # Publication edges
    EdgeType.PUB_VENUE: (NodeType.PUBLICATION, NodeType.VENUE),
    EdgeType.PUB_AUTHOR: (NodeType.PUBLICATION, NodeType.AUTHOR),
    EdgeType.PUB_CO_AUTHOR: (NodeType.PUBLICATION, NodeType.CO_AUTHOR),
    EdgeType.PUB_TRUE_AUTHOR: (NodeType.PUBLICATION, NodeType.AUTHOR),
    EdgeType.PUB_CITES: (NodeType.PUBLICATION, NodeType.PUBLICATION),
    EdgeType.PUB_ORG: (NodeType.PUBLICATION, NodeType.ORGANIZATION),

    # Author edges
    EdgeType.AUTHOR_ORG: (NodeType.AUTHOR, NodeType.ORGANIZATION),
    EdgeType.AUTHOR_PUB: (NodeType.AUTHOR, NodeType.PUBLICATION),
    EdgeType.AUTHOR_CO_AUTHOR: (NodeType.AUTHOR, NodeType.CO_AUTHOR),

    # CoAuthor edges
    EdgeType.CO_AUTHOR_PUB: (NodeType.CO_AUTHOR, NodeType.PUBLICATION),
    EdgeType.CO_AUTHOR_AUTHOR: (NodeType.CO_AUTHOR, NodeType.AUTHOR),
    EdgeType.CO_AUTHOR_ORG: (NodeType.CO_AUTHOR, NodeType.ORGANIZATION),

    # Organization edges
    EdgeType.ORG_PUB: (NodeType.ORGANIZATION, NodeType.PUBLICATION),
    EdgeType.ORG_AUTHOR: (NodeType.ORGANIZATION, NodeType.AUTHOR),
    EdgeType.ORG_CO_AUTHOR: (NodeType.ORGANIZATION, NodeType.CO_AUTHOR),

    # Venue edges
    EdgeType.VENUE_PUB: (NodeType.VENUE, NodeType.PUBLICATION),
}

edge_one_hot = {
    edge_type.value: F.one_hot(torch.tensor(i), num_classes=len(EdgeType)).type(torch.float32)
    for i, edge_type in enumerate(EdgeType)
}
