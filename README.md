
# Master Thesis on Author Disambiguation (AND) on public domain Knowledge Graphs
This repository contains the code and the data used for the experiments in the master thesis on AND on public domain Knowledge Graphs.


## Research Questions
The research questions addressed in this thesis are:
- How can Graph Learning methods be effectively used on the inherently heterogeneous public domain publication
knowledge graphs to disambiguate author names?
- Can relational learning methods be used on the knowledge graph to improve the performance of graph learning
methods for entity resolution tasks?

## Data Sources
- [AMiner](https://www.aminer.cn/aminernetwork)
- [DBLP](https://dblp.org/)
- [WhoIsWho](https://arxiv.org/abs/2302.11848)
- [CiteSeer](http://citeseer.ist.psu.edu/index)
- [META4BUA](https://meta4bua.fokus.fraunhofer.de/datasets?locale=en)

and potentially:
- [Microsoft Academic Graph](https://www.microsoft.com/en-us/research/project/microsoft-academic-graph/)
- [Semantic Scholar](https://api.semanticscholar.org/)
#### Assumptions for data preprocessing and label generation
- The first `n` publications assigned to an author with a large number of publications are probably more likely to be correctly attributed to the author.

## Node Embedding
Generate an embedding for each node type using a Neural Network. 
These embeddings are not learned from the graph structure but from the node features.
Features to encode could include the following:
- Node type (author, publication, venue, keyword, year)
- Node features such as publication metadata (title, abstract), keyword embedding, ...

Furthermore,
- author nodes should be closer in the embedding space if they co-authored a publication.
- publication nodes should be closer in the embedding space if they share the same author(s) and other attributes.
- keywords can be embedded semantically using a transformer model.

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
An embedding for the new author and paper nodes respectively will be generated using the node embedding method.


## Graph Embedding
A Graph Attention Network (GAT) is used to generate embeddings for the nodes in the knowledge graph that encode the structure of the graph.
Based on these embeddings, the similarity between publication nodes can be calculated and used for entity resolution tasks.
