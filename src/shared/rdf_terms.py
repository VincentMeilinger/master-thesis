from enum import Enum


class PublicationRDF(Enum):
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

    CITES = 7


class AuthorRDF(Enum):
    """
    Enumeration of common Author RDF terms used in the datasets.
    """

    ORGANIZATION = 100

    PUBLICATION = 101
