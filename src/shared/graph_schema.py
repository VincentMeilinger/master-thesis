from enum import Enum


class GraphNode(Enum):
    """
    Enumeration of common Graph Node RDF terms used in the datasets.
    """

    PUBLICATION = "Publication"

    AUTHOR = "Author"

    TRUE_AUTHOR = "TrueAuthor"

    ORGANIZATION = "Organization"

    VENUE = "Venue"


class PublicationEdge(Enum):
    """
    Enumeration of common Publication RDF terms used in the datasets.
    """

    IDENTIFIER = 0

    TITLE = 1

    ABSTRACT = 2

    KEYWORD = 3

    VENUE = 4

    YEAR_PUBLISHED = 5

    AUTHOR = 6

    TRUE_AUTHOR = 7

    CITES = 8


class AuthorEdge(Enum):
    """
    Enumeration of common Author RDF terms used in the datasets.
    """

    ORGANIZATION = 100

    PUBLICATION = 101
