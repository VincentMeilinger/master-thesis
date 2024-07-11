
# Master Thesis on Author Disambiguation (AND) on public domain Knowledge Graphs
This repository contains the code and the data used for the experiments in the master thesis on AND on public domain Knowledge Graphs.

## Ideas
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

## Pipeline

<details>
<summary><h3> 
Sentence Transformer Dimensionality Reduction 
</h3></summary>

> `pipeline/emb_dim_reduction.py`

The Sentence Transformer model is used to generate embeddings for the publication titles and abstracts. 
To improve AND performance, the embeddings are reduced to a lower dimensionality using PCA to get the important features and a dense layer with weights initialized using the principal components.
</details>

<details>
<summary><h3> 
Data Preprocessing
</h3></summary>

> `pipeline/preprocess_datasets.py`

Load the publication data from the data sources in the following format: 
 ```json
 [
   {
     "id": "Unique identifier of the publication",
     "title": "Title of the publication",
     "abstract": "Abstract of the publication",
     "authors": [
       {
         "name": "Name of the author",
         "org": "Organization of the author"
       }, ...
     ],
     "venue": "Venue of the publication",
     "year": "Year of the publication",
     "keywords": ["Keyword1", "Keyword2", ...]
   }, ...
 ]
 ```

Standardize and clean the values.
</details>

<details>
<summary><h3> 
Publication Embedding 
</h3></summary>

> `pipeline/embed_datasets.py`

Create embeddings for the publications based on title and abstract in batches. 
The embedding vectors of the publication features are concatenated to form the final embedding.
Save each batch of embeddings (base64 encoded) alongside the respective publication ids in files for later pipeline steps.
The embedding files follow the format:
    ```json
    {
        "id1": "base64 encoded embedding",
        "id2": "base64 encoded embedding",
        ...
    }
    ```

</details>

<details>
<summary><h3> 
Populate the KG database
</h3></summary>
 
> `pipeline/populate_db.py`

Create nodes in the Neo4j database for the publications.
Nodes contain the following properties:
- id
- title
- abstract
- authors
- venue
- year
- keywords
- embedding

</details>
