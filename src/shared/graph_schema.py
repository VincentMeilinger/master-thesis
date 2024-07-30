from enum import Enum


class NodeType(Enum):
    """
    Enumeration of common Graph Node RDF terms used in the datasets.
    """

    PUBLICATION = "Publication"

    AUTHOR = "Author"

    CO_AUTHOR = "CoAuthor"

    TRUE_AUTHOR = "TrueAuthor"

    ORGANIZATION = "Organization"

    VENUE = "Venue"



class EdgeType(Enum):
    pass

class SimilarityEdge(EdgeType):
    SIM_ORG = "SimilarOrganization"
    SIM_VENUE = "SimilarVenue"
    SIM_TITLE = "SimilarTitle"
    SIM_ABSTRACT = "SimilarAbstract"
    SIM_KEYWORD = "SimilarKeyword"
    SIM_YEAR = "SimilarYear"
    SIM_AUTHOR = "SimilarAuthor"


class PublicationEdge(EdgeType):
    """
    Enumeration of common Publication RDF terms used in the datasets.
    """
    TITLE = "SimilarTitle"
    ABSTRACT = "SimilarAbstract"
    KEYWORD = "SimilarKeyword"
    VENUE = "Venue"
    YEAR_PUBLISHED = "YearPublished"
    AUTHOR = "Author"
    CONTRIBUTOR = "Contributor"
    TRUE_AUTHOR = "TrueAuthor"
    CITES = "Cites"


class AuthorEdge(EdgeType):
    """
    Enumeration of common Author RDF terms used in the datasets.
    """
    ORGANIZATION = "Organization"
    PUBLICATION = "Publication"
