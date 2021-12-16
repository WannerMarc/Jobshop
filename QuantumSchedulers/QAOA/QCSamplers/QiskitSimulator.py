from QuantumSchedulers.QAOA.QAOA import QCSampler
from qiskit import Aer, transpile


class QiskitSimulator(QCSampler):
    def __init__(self, simulator_type='qasm_simulator'):
        super().__init__()
        self._simulator_type = simulator_type

    def sample_qc(self, quantum_circuit, num_reads):
        self._backend = Aer.get_backend(self._simulator_type)
        qobj = transpile(quantum_circuit, self._backend)
        counts = self._backend.run(qobj, seed_simulator=self._seed, shots=num_reads).result().get_counts()
        return counts

    def get_name(self):
        return "QISKITSIMULATOR_"+self._simulator_type

