from enum import Enum


class RdfTerms(Enum):
    """
    Enumeration of common RDF terms used in the datasets. Contains the URI of the term as the value of the enum.
    Encompasses terms from the Dublin Core Metadata Initiative (DCMI) and the Friend of a Friend (FOAF).
    """

    # DCMI

    IDENTIFIER = 'http://purl.org/dc/terms/identifier'
    '''An unambiguous reference to the resource within a given context.'''

    CREATOR = 'http://purl.org/dc/terms/creator'
    '''An entity responsible for making the resource.'''

    TITLE = 'http://purl.org/dc/terms/title'
    '''A name given to the resource.'''

    ABSTRACT = 'http://purl.org/dc/terms/abstract'
    '''A summary of the resource.'''

    PUBLISHER = 'http://purl.org/dc/terms/publisher'
    '''An entity responsible for making the resource available.'''

    CITED_BY = 'http://purl.org/dc/terms/isReferencedBy'
    '''A related resource that is referenced, cited, or otherwise pointed to by the described resource.'''

    CITES = 'http://purl.org/dc/terms/references'
    '''A related resource that references, cites, or otherwise points to the described resource.'''

    DATE_CREATED = 'http://purl.org/dc/terms/created'
    '''Date of creation of the resource.'''

    PART_OF = 'http://purl.org/dc/terms/isPartOf'
    '''A related resource in which the described resource is physically or logically included.'''

    # FOAF

    PERSON = 'http://xmlns.com/foaf/0.1/Person'
    '''A person.'''

    ORGANIZATION = 'http://xmlns.com/foaf/0.1/Organization'
    '''An organization.'''

    NAME = 'http://xmlns.com/foaf/0.1/name'
    '''A name for some thing.'''

    KNOWS = 'http://xmlns.com/foaf/0.1/knows'
    '''A person known by this person (indicating some level of reciprocated interaction between the parties).'''
