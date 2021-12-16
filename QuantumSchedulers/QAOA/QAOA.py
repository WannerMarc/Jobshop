from abc import ABCMeta, abstractmethod
import sys
sys.path.append('../..')
import time


class Preprocessor(metaclass=ABCMeta):
    def __init__(self):
        # initialize member variable with some value if not needed for preprocessing e.g. self._hamiltonian = 0
        # unfortunately have to extend qaoa_data with None entries for each different preprocess approach
        # but this allows for preprocessing that affects different parts of the QAOA pipeline
        self._hamiltonian = None
        self._scheduling_data = None
        self._qaoa_data: dict = {"CONTINUOUS_SOLUTION": None}
        self._qaoa_data_name = None
        self._time = 0

    def get_preprocess_data(self, hamiltonian=None, scheduling_data=None):
        start = time.time()
        if self._qaoa_data[self._qaoa_data_name] is None:
            assert not (hamiltonian is None and self._hamiltonian is None)
            assert not (scheduling_data is None and self._scheduling_data is None)
            self.preprocess(hamiltonian, scheduling_data)
        end = time.time()
        self._time = end - start
        return self._qaoa_data

    @abstractmethod
    def preprocess(self, hamiltonian=None, scheduling_data=None):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def get_time(self):
        return self._time


class CircuitBuilder(metaclass=ABCMeta):
    def __init__(self):
        self._quantum_circuit = None
        self._theta = None
        self._bqm = None
        self._num_qubits = None
        self._qaoa_data: dict = None
        self._time = 0
        self._nreps = 0

    def get_quantum_circuit(self, theta=None, bqm=None, num_qubits=None, qaoa_data=None):
        start = time.time()
        if qaoa_data is not None:
            self.set_preprocess_data(qaoa_data)
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
        end = time.time()
        self._time += end - start
        self._nreps += 1
        return self._quantum_circuit

    def set_bqm(self, bqm, num_qubits):
        self._bqm = bqm
        self._num_qubits = num_qubits

    def set_preprocess_data(self, qaoa_data: dict):
        self._qaoa_data = qaoa_data

    def get_time(self):
        return self._time/self._nreps

    def reset_time(self):
        self._time = 0
        self._nreps = 0

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
        self._time = 0
        self._nreps = 0

    def get_counts(self, quantum_circuit, num_reads):
        start = time.time()
        counts = self.sample_qc(quantum_circuit, num_reads)
        end = time.time()
        self._time += end - start
        self._nreps += 1
        return counts

    @abstractmethod
    def sample_qc(self, quantum_circuit, num_reads):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def get_seed(self):
        return self._seed

    def get_time(self):
        return self._time/self._nreps

    def reset_time(self):
        self._time = 0
        self._nreps = 0

    def get_nreps(self):
        return self._nreps


class ThetaOptimizer(metaclass=ABCMeta):
    def __init__(self):
        # initialize member variable with some value if not needed for preprocessing e.g. self._qc = 0
        self._theta = None
        self._circuit_builder = None
        self._qc_sampler = None
        self._num_reads = None
        self._hamiltonian = None
        self._expected_energy = None
        self._time = 0

    def get_theta(self, hamiltonian, theta_init, num_reads: int, circuit_builder=None, qc_sampler=None):
        start = time.time()
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
        end = time.time()
        self._time = end - start
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

    def get_time(self):
        return self._time


class Postprocessor(metaclass=ABCMeta):
    def __init__(self):
        self._postprocessed_counts: dict = None
        self._postprocessing_input: dict = None
        self._time = 0

    def get_postprocessed_data(self, postprocessing_input: dict = None):
        start = time.time()
        if self._postprocessed_counts is None:
            assert postprocessing_input is not None
            self.postprocess(postprocessing_input)
        end = time.time()
        self._time = end - start
        return self._postprocessed_counts

    @abstractmethod
    def postprocess(self, postprocessing_input: dict):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def get_time(self):
        return self._time






