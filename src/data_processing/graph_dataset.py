import pandas as pd


class GraphDataset:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    valid: pd.DataFrame

    def __init__(self, name: str, train: pd.DataFrame, test: pd.DataFrame, valid: pd.DataFrame):
        self.name = name
        self.train = train
        self.test = test
        self.valid = valid

    def count_distinct_entities(self, filter: str = None, split: str = 'all'):
        # Initial empty Series to collect entities
        entities = pd.Series(dtype=str)

        # Gather entities from the specified splits
        if split == 'train' or split == 'all':
            entities = pd.concat([self.train['h'], self.train['t']])
        if split == 'test' or split == 'all':
            entities = pd.concat([entities, self.test['h'], self.test['t']])
        if split == 'valid' or split == 'all':
            entities = pd.concat([entities, self.valid['h'], self.valid['t']])

        # Drop duplicates to get distinct entities
        distinct_entities = entities.drop_duplicates()

        # Filter the distinct entity strings based on the filter
        if filter:
            distinct_entities = distinct_entities[distinct_entities.str.contains(filter)]

        return len(distinct_entities)

    def count_citations(self, filter: str = None):
        relations = pd.concat([self.train['r'], self.test['r'], self.valid['r']])
        if filter:
            relations = relations[relations.str.contains(filter)]
        return relations.size
