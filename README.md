
# Master Thesis on Author Disambiguation (AND) on public domain Knowledge Graphs
This repository contains the code and the data used for the experiments in the master thesis on AND on public domain Knowledge Graphs.

## Idea
- **The Knowledge Graph:** Construct a KG from all publications and authors in the data. Prior knowledge can be incorporated by creating "unbreakable" links between authors and their publications.
- **Node Embedding:** Generate embeddings for the nodes in the KG.
  - **Paper Embedding:** Train a link prediction model (e.g. DistMult) using a GAT layer to incorporate structural information to create embeddings for publications based on co-author network, citation network, ...
  - **Author Embedding:** Use a GAT to create embeddings for authors based on co-author network, publication network, ...
- **Step 1:** Create links between publications and their potential authors using the translational model. Sever the links between authors and publications that are likely not correct.
- **Step 2:** Create links between authors that are likely identical.

## Research Questions
The research questions addressed in this thesis are:
- How can Graph Learning methods be effectively used on the inherently heterogeneous public domain publication
knowledge graphs to disambiguate author names?
- Can Relational Learning methods be used on the knowledge graph for publication linkage (KGC) to improve the performance of graph learning
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




## Graph Embedding
A Graph Attention Network (GAT) is used to generate embeddings for the nodes in the knowledge graph that encode the structure of the graph.


## Pipeline

**TODO:** This section will be updated when the pipeline design is finalized.
