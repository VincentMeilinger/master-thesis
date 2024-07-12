
class Dataset:
    name: str = 'name'

    def __init__(self):
        pass

    @staticmethod
    def parse_dict():
        """ Parse a dataset and return as a dictionary. """
        raise NotImplementedError("Method _parse_dict not implemented.")

    @staticmethod
    def parse_data():
        """ Parse the dataset. """
        raise NotImplementedError("Method parse_data not implemented.")

    @staticmethod
    def parse_valid():
        """ Parse the validation dataset. """
        raise NotImplementedError("Method parse_valid not implemented.")

    @staticmethod
    def parse_train():
        """ Parse the training dataset. """
        raise NotImplementedError("Method parse_train not implemented.")

    @staticmethod
    def parse_graph():
        """ Parse the graph dataset. """
        raise NotImplementedError("Method parse_graph not implemented.")
