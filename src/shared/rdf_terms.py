from enum import Enum


class RdfTerms(Enum):
    """
    Enumeration of common RDF terms used in the datasets. Contains the URI of the term as the value of the enum.
    Encompasses terms from the Dublin Core Metadata Initiative (DCMI) and the Friend of a Friend (FOAF).
    """

    # DCMI

    IDENTIFIER = 0
    '''An unambiguous reference to the resource within a given context.'''

    CREATOR = 1
    '''An entity responsible for making the resource.'''

    TITLE = 2
    '''A name given to the resource.'''

    ABSTRACT = 3
    '''A summary of the resource.'''

    PUBLISHER = 4
    '''An entity responsible for making the resource available.'''

    CITED_BY = 5
    '''A related resource that is referenced, cited, or otherwise pointed to by the described resource.'''

    CITES = 6
    '''A related resource that references, cites, or otherwise points to the described resource.'''

    DATE_CREATED = 7
    '''Date of creation of the resource.'''

    PART_OF = 8
    '''A related resource in which the described resource is physically or logically included.'''

    # FOAF

    PERSON = 9
    '''A person.'''

    ORGANIZATION = 10
    '''An organization.'''

    NAME = 11
    '''A name for some thing.'''

    KNOWS = 12
    '''A person known by this person (indicating some level of reciprocated interaction between the parties).'''

    # Other

    KEYWORD = 13
    '''A keyword or tag for the resource.'''

    VENUE = 14
    '''The venue where the resource was published or presented.'''

    YEAR = 15
    '''The year of publication or presentation of the resource.'''

    def uri(self):
        # Encode all cases using integer values
        return RDF_TERM_URIS[self]

    @staticmethod
    def from_uri(uri: str):
        for term in RdfTerms:
            if RDF_TERM_URIS[term] == uri:
                return term
        return None


RDF_TERM_URIS = {
    RdfTerms.IDENTIFIER: 'http://purl.org/dc/terms/identifier',
    RdfTerms.CREATOR: 'http://purl.org/dc/terms/creator',
    RdfTerms.TITLE: 'http://purl.org/dc/terms/title',
    RdfTerms.ABSTRACT: 'http://purl.org/dc/terms/abstract',
    RdfTerms.PUBLISHER: 'http://purl.org/dc/terms/publisher',
    RdfTerms.CITED_BY: 'http://purl.org/dc/terms/isReferencedBy',
    RdfTerms.CITES: 'http://purl.org/dc/terms/references',
    RdfTerms.DATE_CREATED: 'http://purl.org/dc/terms/created',
    RdfTerms.PART_OF: 'http://purl.org/dc/terms/isPartOf',
    RdfTerms.PERSON: 'http://xmlns.com/foaf/0.1/Person',
    RdfTerms.ORGANIZATION: 'http://xmlns.com/foaf/0.1/Organization',
    RdfTerms.NAME: 'http://xmlns.com/foaf/0.1/name',
    RdfTerms.KNOWS: 'http://xmlns.com/foaf/0.1/knows',
    RdfTerms.KEYWORD: 'KEYWORD URI',
    RdfTerms.VENUE: 'VENUE URI',
    RdfTerms.YEAR: 'YEAR URI'
}
