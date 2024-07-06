import configparser
import os


class TransformerDimReductionState:
    def __init__(self):
        self.state = 'not_started'


class CreateNodesState:
    def __init__(self):
        self.create_publication_nodes_state = 'not_started'
        self.add_true_authors_state = 'not_started'


class EmbedNodesState:
    def __init__(self):
        self.state = 'not_started'


class CreateEdgesState:
    def __init__(self):
        self.state = 'not_started'


class TrainGraphModelState:
    def __init__(self):
        self.state = 'not_started'


class EvaluateGraphModelState:
    def __init__(self):
        self.state = 'not_started'


class PipelineState:
    def __init__(self, run_id: str, run_path: str):
        self.run_id = run_id
        self.run_path = run_path

        self.transformer_dim_reduction = TransformerDimReductionState()
        self.create_nodes = CreateNodesState()
        self.embed_nodes = EmbedNodesState()
        self.create_edges = CreateEdgesState()
        self.train_graph_model = TrainGraphModelState()
        self.evaluate_graph_model = EvaluateGraphModelState()

        try:
            if not os.path.exists(os.path.join(self.run_path, 'pipeline_state.ini')):
                raise Exception("Run configuration file not found.")
            self.load()
        except Exception as e:
            self.save()

    def load(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(self.run_path, 'run_config.ini'))

        self.run_id = config.get('run', 'run_id', fallback='not_started')
        self.run_path = config.get('run', 'run_path', fallback='not_started')
        self.transformer_dim_reduction.state = config.get('transformer_dim_reduction', 'state', fallback='not_started')
        self.create_nodes.create_publication_nodes_state = config.get('create_nodes', 'create_publication_nodes_state', fallback='not_started')
        self.create_nodes.add_true_authors_state = config.get('create_nodes', 'add_true_authors_state', fallback='not_started')
        self.embed_nodes.state = config.get('embed_nodes', 'state', fallback='not_started')
        self.create_edges.state = config.get('create_edges', 'state', fallback='not_started')
        self.train_graph_model.state = config.get('train_graph_model', 'state', fallback='not_started')
        self.evaluate_graph_model.state = config.get('evaluate_graph_model', 'state', fallback='not_started')

    def save(self):
        config = configparser.ConfigParser()
        config.add_section('run')
        config.add_section('transformer_dim_reduction')
        config.add_section('create_nodes')
        config.add_section('embed_nodes')
        config.add_section('create_edges')
        config.add_section('train_graph_model')
        config.add_section('evaluate_graph_model')

        config.set('run', 'run_id', self.run_id)
        config.set('run', 'run_path', self.run_path)
        config['transformer_dim_reduction'] = self.transformer_dim_reduction.__dict__
        config['create_nodes'] = self.create_nodes.__dict__
        config['embed_nodes'] = self.embed_nodes.__dict__
        config['create_edges'] = self.create_edges.__dict__
        config['train_graph_model'] = self.train_graph_model.__dict__
        config['evaluate_graph_model'] = self.evaluate_graph_model.__dict__

        with open(os.path.join(self.run_path, 'pipeline_state.ini'), 'w') as configfile:
            config.write(configfile)

    def reset(self):
        os.remove(os.path.join(self.run_path, 'pipeline_state.ini'))

    def to_string(self):
        return self.__dict__
