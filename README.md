
# Master Thesis on Author Disambiguation (AND) on public domain Knowledge Graphs
This repository contains the code and the data used for the experiments in the master thesis on AND on public domain Knowledge Graphs.

## Research Questions
The research questions addressed in this thesis are:

**Primary Research Questions:**
- How can Graph Learning methods be effectively used on public domain publication knowledge graphs to disambiguate author names?
  - Can Graph Attention Networks (GAT) be used to generate embeddings for the nodes in the knowledge graph? 
  - What learning methods can be used to generate graph embeddings that encode the structure of the knowledge graph?
**Secondary Research Questions:**
- Can Link Prediction be used to enhance AND performance?
  - Is Relational Learning viable to improve the performance entity resolution?
  - What is the effect of incorporating prior knowledge in the form of knowledge graph links into the AND process?

## Data Sources
- [WhoIsWho](https://arxiv.org/abs/2302.11848)
- [AMiner](https://www.aminer.cn/aminernetwork)
- [DBLP](https://dblp.org/)
- [CiteSeer](http://citeseer.ist.psu.edu/index)


## Graph Embedding
A Graph Attention Network (GAT) is used to generate embeddings for the nodes in the knowledge graph that encode the structure of the graph.


## Pipeline

### Files and Modules

- `main.py`: Entry point for the pipeline execution. It provides CLI options to manage different stages of the pipeline.
- `pipeline_config.py`: Configuration file specifying model parameters and thresholds for the pipelines.
- `create_nodes.py`: Creates and merges publication nodes into the Neo4j database with embeddings for titles, abstracts, venues, and organizations.
- `link_nodes.py`: Links nodes based on attribute similarity (cosine similarity) and co-author relationships.
- `prediction.py`: Predicts and resolves ambiguities between nodes (authors) using a Siamese neural network.
- `transformer_dim_reduction.py`: Prepares a transformer model with reduced embedding dimensions (for node attributes) using PCA for better performance.

### Prerequisites

- **Neo4j Database**: Ensure Neo4j is installed and running. Update `pipeline_config.py` and environment variables in `src.shared.config.py` with the appropriate database settings.
- **Python Dependencies**: Install the required Python libraries:
  ```bash
  pip install -r requirements.txt
  ```
- **Hardware**: GPU recommended for faster model training and inference. CUDA is required for GPU support.
- **Datasets**: Download and extract the WhoIsWho dataset and place it in the `data` directory. The dataset should be in the following format:
  ```
  data/
  ├── datasets/
      ├── IND-WhoIsWho.json
      ├── ...
  ```
### Running the Pipeline
1. **Delete the Database** (Optional): If needed, clear the database before initializing the pipeline:
  ```bash
  python main.py --delete_db
  ```
2. **Transformer Model Modification**: Add a dense layer for dimensionality reduction:
  ```bash
  python main.py --prepare_pipeline
  ```
3. **Create and Link Nodes**: Establish relationships between nodes based on attribute similarity and co-author overlap:
  ```bash
  python main.py --create_nodes
  ```
4. **Disambiguation Pipeline**: Resolve ambiguities between authors using a trained Siamese neural network:
  ```bash
  python main.py --prediction
  ```

### Configuration Details

Modify pipeline_config.py for customization:

- Transformer Settings
  - _base_model_: Pretrained transformer model to use (e.g., jordyvl/scibert_scivocab_uncased_sentence_transformer).
  - _reduced_dim_: Target dimensionality for embeddings.
  - _num_pca_samples_: Number of samples for PCA training.
    
- Linking Thresholds
  - _link_title_threshold_: Threshold for title similarity.
  - _link_abstract_threshold_: Threshold for abstract similarity.
  - Additional thresholds for venue and organization attributes.
    
- GAT Model Parameters
  - _hidden_channels_: Number of hidden dimensions for GAT layers.
  - _num_heads_: Number of attention heads for GAT layers.
  - _classifier_dropout_: Dropout rate for the classifier.