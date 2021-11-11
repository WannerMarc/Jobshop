from abc import ABCMeta, abstractmethod
import sys
sys.path.append('../..')


class Preprocessor(metaclass=ABCMeta):
    def __init__(self):
        # initialize member variable with some value if not needed for preprocessing e.g. self._hamiltonian = 0
        self._hamiltonian = None
        self._scheduling_data = None
        self._qaoa_data: dict = None

    def get_preprocess_data(self, hamiltonian=None, scheduling_data=None):
        if self._qaoa_data is None:
            assert not (hamiltonian is None and self._hamiltonian is None)
            assert not (scheduling_data is None and self._scheduling_data is None)
            self.preprocess(hamiltonian, scheduling_data)

        return self._qaoa_data

    @abstractmethod
    def preprocess(self, hamiltonian=None, scheduling_data=None):
        pass

    @abstractmethod
    def get_name(self):
        pass


class CircuitBuilder(metaclass=ABCMeta):
    def __init__(self):
        self._quantum_circuit = None
        self._theta = None
        self._bqm = None
        self._num_qubits = None

    def get_quantum_circuit(self, theta=None, bqm=None, num_qubits=None):
        if self._quantum_circuit is None:
            assert theta is not None
            assert not(self._bqm is None and (bqm is None or num_qubits is None))
            if bqm is None:
                bqm = self._bqm
                num_qubits = self._num_qubits
            self.build_quantum_circuit(theta, bqm, num_qubits)
        elif theta is not None:
            if bqm is None:
                bqm = self._bqm
                num_qubits = self._num_qubits
            else:
                assert num_qubits is not None
            self.build_quantum_circuit(theta, bqm, num_qubits)

        return self._quantum_circuit

    def set_bqm(self, bqm, num_qubits):
        self._bqm = bqm
        self._num_qubits = num_qubits

    @abstractmethod
    def build_quantum_circuit(self, theta, bqm, num_qubits: int):
        pass

    @abstractmethod
    def get_name(self):
        pass


class QCSampler(metaclass=ABCMeta):
    def __init__(self, seed_simulator=937162211):
        self._backend = None
        self._seed = seed_simulator

    @abstractmethod
    def sample_qc(self, quantum_circuit, num_reads):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def get_seed(self):
        return self._seed


class ThetaOptimizer(metaclass=ABCMeta):
    def __init__(self):
        # initialize member variable with some value if not needed for preprocessing e.g. self._qc = 0
        self._theta = None
        self._circuit_builder = None
        self._qc_sampler = None
        self._num_reads = None
        self._hamiltonian = None
        self._expected_energy = None

    def get_theta(self, hamiltonian, theta_init, num_reads: int, circuit_builder=None, qc_sampler=None):
        if self._theta is None:
            assert not (circuit_builder is None and self._circuit_builder is None)
            assert not (qc_sampler is None and self._qc_sampler is None)
            self.optimize_theta(circuit_builder, qc_sampler, num_reads, hamiltonian, theta_init)
        elif theta_init is not None:
            cb = circuit_builder
            qs = qc_sampler
            if cb is None:
                cb = self._circuit_builder
            if qs is None:
                qs = self._qc_sampler
            self.optimize_theta(cb, qs, num_reads, hamiltonian, theta_init)

        return self._theta

    def get_expected_energy(self):
        return self._expected_energy

    @abstractmethod
    def optimize_theta(self, circuit_builder: CircuitBuilder, qc_sampler: QCSampler, num_reads: int, hamiltonian,
                       theta_init):
        pass

    @abstractmethod
    def get_name(self):
        pass


class Postprocessor(metaclass=ABCMeta):
    def __init__(self):
        self._postprocessed_counts: dict = None
        self._postprocessing_input: dict = None

    def get_postprocessed_data(self, postprocessing_input: dict = None):
        if self._postprocessed_counts is None:
            assert postprocessing_input is not None
            self.postprocess(postprocessing_input)

        return self._postprocessed_counts

    @abstractmethod
    def postprocess(self, postprocessing_input: dict):
        pass

    @abstractmethod
    def get_name(self):
        pass






