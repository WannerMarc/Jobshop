from QuantumSchedulers.QAOA.QAOA import QCSampler


class LazySampler(QCSampler):
    def __init__(self, simulator_type='qasm_simulator'):
        super().__init__()
        self._simulator_type = simulator_type
        self._counts = None
        self._expected_energy = None

    def sample_qc(self, quantum_circuit, num_reads):
        return self._counts

    def get_name(self):
        return "LAZYSAMPLER_"+self._simulator_type

    def set_counts(self, counts):
        self._counts = counts

    def set_expected_energy(self, expected_energy):
        self._expected_energy = expected_energy

    def get_expected_energy(self):
        return self._expected_energy



