import json
from QuantumSchedulers.QAOA.QAOA import Preprocessor


class LazyPreprocessor(Preprocessor):
    def __init__(self, filename):
        super().__init__()
        with open(filename) as json_file:
            result_data = json.load(json_file)
            self._qaoa_data = result_data["QAOA_DATA"]

    def preprocess(self, hamiltonian=None, scheduling_data=None):
        return

    def get_name(self):
        return "LAZY_PREPROCESSOR"