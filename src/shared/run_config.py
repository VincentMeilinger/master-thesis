import configparser
import os


class TransformerDimReductionConfig:
    def __init__(self):
        self.base_model = "all-mpnet-base-v2"
        self.reduced_dim = 16
        self.num_pca_samples = 10000


class PopulateDBConfig:
    def __init__(self):
        self.db_name = "knowledge_graph"
        self.max_nodes = 1000
        self.max_seq_len = 256


class EmbedDatasetsConfig:
    def __init__(self):
        self.transformer_model = "./data/models/all-mpnet-base-v2-16dim"
        self.batch_size = 10000


class RunConfig:
    def __init__(self, run_path: str):
        self.run_path = run_path
        self.transformer_dim_reduction = TransformerDimReductionConfig()
        self.populate_db = PopulateDBConfig()
        self.embed_datasets = EmbedDatasetsConfig()

        try:
            self.load()
        except Exception as e:
            self.save()
            raise Exception(
                "Run configuration file not found. A new config was created. Configure the pipeline and retry."
            )

    def load(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(self.run_path, 'run_config.ini'))

        self.transformer_dim_reduction.model_name = config.get('transformer_dim_reduction', 'base_model')
        self.transformer_dim_reduction.reduced_dim = config.getint('transformer_dim_reduction', 'reduced_dim')
        self.transformer_dim_reduction.num_pca_samples = config.getint('transformer_dim_reduction', 'num_pca_samples')

        self.populate_db.db_name = config.get('populate_db', 'db_name')
        self.populate_db.max_nodes = config.getint('populate_db', 'max_nodes')
        self.populate_db.max_seq_len = config.getint('populate_db', 'max_seq_len')

        self.embed_datasets.transformer_model = config.get('embed_datasets', 'transformer_model')
        self.embed_datasets.batch_size = config.getint('embed_datasets', 'batch_size')

    def save(self):
        config = configparser.ConfigParser()
        config.add_section('transformer_dim_reduction')
        config.add_section('populate_db')
        config.add_section('embed_datasets')
        config.add_section('train_graph_model')
        config.add_section('evaluate_graph_model')
        config['transformer_dim_reduction'] = self.transformer_dim_reduction.__dict__
        config['populate_db'] = self.populate_db.__dict__
        config['embed_datasets'] = self.embed_datasets.__dict__

        with open(os.path.join(self.run_path, 'run_config.ini'), 'w') as configfile:
            config.write(configfile)

    def to_string(self):
        return self.__dict__
