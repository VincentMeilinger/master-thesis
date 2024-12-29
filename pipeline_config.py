from src.shared.graph_schema import NodeType, EdgeType

config = {
  "database": "small-graph",
  "transformer_dim_reduction": {
    'base_model': 'jordyvl/scibert_scivocab_uncased_sentence_transformer',
    'reduced_dim': 32,
    'num_pca_samples': 10000,
  },
  "link_node": {
    "co_author_overlap_threshold": 0.25,
    "link_title_threshold": 0.7,
    "link_abstract_threshold": 0.7,
    "link_venue_threshold": 0.7,
    "link_org_threshold": 0.9,

    "link_title_k": 8,
    "link_abstract_k": 8,
    "link_venue_k": 8,
    "link_org_k": 8,
  },
  "gat_list": {
    EdgeType.SIM_ABSTRACT: "./notebooks/data/results/homogeneous (abstract) full_emb linear_layer dropout 32h 8out/gat_encoder.pt",
    EdgeType.SIM_AUTHOR: "./notebooks/data/results/homogeneous (similar co-authors) full_emb linear_layer dropout small_graph low_dim/gat_encoder.pt",
    EdgeType.SIM_ORG: "./notebooks/data/results/homogeneous (org) full_emb linear_layer dropout 32h 8out/gat_encoder.pt",
    EdgeType.SAME_AUTHOR: "./notebooks/data/results/homogeneous (same author) full_emb linear_layer dropout low_dim/gat_encoder.pt"
  },
  "gat_specs": {
    'max_hops': 2,
    'model_node_feature': 'feature_vec',  # Node feature to use for GAT encoder
    'hidden_channels': 32,
    'out_channels': 8,
    'num_heads': 8,
    'classifier_in_channels': 4 * 8,
    'classifier_hidden_channels': 16,
    'classifier_out_channels': 8,
    'classifier_dropout': 0.3,
  },
  "node_spec": [
    NodeType.PUBLICATION
  ],
  "edge_spec": [
    EdgeType.SIM_ABSTRACT,
    EdgeType.SIM_AUTHOR,
    EdgeType.SIM_ORG,
    EdgeType.SAME_AUTHOR,
  ],
  "embedding_net_path": "./notebooks/data/results/classifier full_emb (abstract, org, sim_author, same_author edges) low dim 2 layers/embedding_net.pt",
  "margin": 0.3,
}