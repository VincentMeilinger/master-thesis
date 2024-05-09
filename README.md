
# Master Thesis on Author Disambiguation (AND) on public domain Knowledge Graphs
This repository contains the code and the data used for the experiments in the master thesis on AND on public domain Knowledge Graphs.


## Research Questions
The research questions addressed in this thesis are:
- How can Graph Learning methods be effectively used on the inherently heterogeneous public domain publication
knowledge graphs to disambiguate author names?
- Can relational learning methods be used on the knowledge graph to improve the performance of graph learning
methods for entity resolution tasks?

## Data Sources
- [Berlin University Alliance](https://meta4bua.fokus.fraunhofer.de/datasets?locale=en)
- ...

## Knowledge Graph
The constructed knowledge graph contains the following node types:
- Author
- Publication
- Venue
- Keyword
- Year

and the following link types:
- (Publication) - **publishedIn** - (Venue)
- (Publication) - **cites** - (Publication)
- (Author) - **wrote** - (Publication)
- (Venue) - **published** - (Publication)
- (Publication) - **writtenIn** - (Year)
- (Publication) - **references** - (Keyword)
- (Author) - **fieldOfStudy** - (Keyword)
- (Author) - **collaboratesWith** - (Author)

#### When a new publication is added to the KG, ...
... the publication is linked to the existing nodes (Author, Venue, Year, Keyword) based on the publication metadata.
The author will be initialized as a new node.
The publication will be linked to the author with the link type **wrote**.
The author will be linked to the keywords with the link type **fieldOfStudy**.
The author will be linked to the venue with the link type **published**.
An embedding for the new author and paper nodes respectively will be generated using the graph learning method.



