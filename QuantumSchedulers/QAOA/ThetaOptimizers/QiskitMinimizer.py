from scipy.optimize import minimize
from QuantumSchedulers.QAOA.QAOA import ThetaOptimizer, QCSampler, CircuitBuilder
import numpy as np


class QiskitMinimizer(ThetaOptimizer):
    def __init__(self, method: str):
        super().__init__()
        self._method = method

    def optimize_theta(self, circuit_builder: CircuitBuilder, qc_sampler: QCSampler, num_reads: int, hamiltonian,
                       theta, verbose=False):
        self._circuit_builder = circuit_builder
        self._qc_sampler = qc_sampler
        self._num_reads = num_reads
        self._hamiltonian = hamiltonian

        expectation = get_expectation(self._hamiltonian, self._circuit_builder, self._qc_sampler, self._num_reads)
        res = minimize(expectation, theta, method=self._method)
        self._theta = res.x
        self._expected_energy = res.fun
        if verbose:
            print(res)

    def get_name(self):
        return "QISKITMINIMIZER_" + self._method


def key_to_vector(key: str):
    return np.array([int(c) for c in key], dtype=np.int)


def expected_value(counts: dict, num_reads: int, hamiltonian):
    F = 0
    for bitstring, counts in counts.items():
        x = key_to_vector(bitstring)
        F += counts*np.dot(np.dot(x, hamiltonian), x)
    return F/num_reads


def get_expectation(hamiltonian, builder, sampler, num_reads):
    def execute_circ(theta):
        qc = builder.get_quantum_circuit(theta)
        counts = sampler.sample_qc(qc, num_reads)
        return expected_value(counts, num_reads, hamiltonian)

    return execute_circ
