import configparser
import os


class TransformerDimReductionConfig:
    def __init__(self):
        self.base_model = "all-mpnet-base-v2"
        self.reduced_dim = 16
        self.num_pca_samples = 10000


class CreateNodesConfig:
    def __init__(self):
        self.db_name = "knowledge_graph"
        self.max_nodes = 1000
        self.max_seq_len = 256


class EmbedDatasetsConfig:
    def __init__(self):
        self.transformer_model = "./data/models/all-mpnet-base-v2-16dim"
        self.batch_size = 10000


class CreateEdgesConfig:
    def __init__(self):
        self.batch_size = 10000
        self.similarity_threshold = 0.95
        self.k_nearest_limit = 10


class RunConfig:
    def __init__(self, run_path: str):
        self.run_path = run_path
        self.transformer_dim_reduction = TransformerDimReductionConfig()
        self.create_nodes = CreateNodesConfig()
        self.embed_datasets = EmbedDatasetsConfig()
        self.create_edges = CreateEdgesConfig()

        try:
            if not os.path.exists(os.path.join(self.run_path, 'run_config.ini')):
                raise Exception("Run configuration file not found.")
            self.load()
        except Exception as e:
            self.save()
            raise Exception(
                "Run configuration file not found. A new config was created. Configure the pipeline and retry."
            )

    def load(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(self.run_path, 'run_config.ini'))

        self.transformer_dim_reduction.base_model = config.get('transformer_dim_reduction', 'base_model', fallback='all-mpnet-base-v2')
        self.transformer_dim_reduction.reduced_dim = config.getint('transformer_dim_reduction', 'reduced_dim', fallback=16)
        self.transformer_dim_reduction.num_pca_samples = config.getint('transformer_dim_reduction', 'num_pca_samples', fallback=10000)

        self.create_nodes.db_name = config.get('populate_db', 'db_name', fallback='knowledge_graph')
        self.create_nodes.max_nodes = config.getint('populate_db', 'max_nodes', fallback=0)
        self.create_nodes.max_seq_len = config.getint('populate_db', 'max_seq_len', fallback=256)

        self.embed_datasets.transformer_model = config.get('embed_datasets', 'transformer_model', fallback='./data/models/all-mpnet-base-v2-16dim')
        self.embed_datasets.batch_size = config.getint('embed_datasets', 'batch_size', fallback=1000)

        self.create_edges.batch_size = config.getint('create_edges', 'batch_size', fallback=1000)
        self.create_edges.similarity_threshold = config.getfloat('create_edges', 'similarity_threshold', fallback=0.95)
        self.create_edges.k_nearest_limit = config.getint('create_edges', 'k_nearest_limit', fallback=10)

    def save(self):
        config = configparser.ConfigParser()
        config.add_section('transformer_dim_reduction')
        config.add_section('populate_db')
        config.add_section('embed_datasets')
        config.add_section('create_edges')
        config.add_section('train_graph_model')
        config.add_section('evaluate_graph_model')
        config['transformer_dim_reduction'] = self.transformer_dim_reduction.__dict__
        config['populate_db'] = self.create_nodes.__dict__
        config['embed_datasets'] = self.embed_datasets.__dict__
        config['create_edges'] = self.create_edges.__dict__

        with open(os.path.join(self.run_path, 'run_config.ini'), 'w') as configfile:
            config.write(configfile)

    def to_string(self):
        return self.__dict__
