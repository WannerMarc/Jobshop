from QuantumSchedulers.QAOA.QAOA import CircuitBuilder
from QuantumSchedulers.QAOA.CircuitBuilders.QuboCircuitBuilder import u_problem_dense, u_problem_sparse
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import numpy as np


class WarmstartCircuitBuilder(CircuitBuilder):
    def __init__(self, epsilon=0):
        super().__init__()
        self._epsilon = epsilon

    def build_quantum_circuit(self, theta, bqm, num_qubits: int):
        self._theta = theta
        self._bqm = bqm
        self._num_qubits = num_qubits
        p = int(len(theta) / 2)
        self._quantum_circuit = QuantumCircuit(num_qubits)
        betas = theta[:p]
        gammas = theta[p:]
        u_problem = u_problem_dense

        if type(bqm).__name__ is 'BinaryQuadraticModel':
            u_problem = u_problem_sparse

        assert self.preprocessing_compatible(), "Preprocessing is not compatible with " + self.get_name()

        warmstart_thetas = [2 * np.arcsin(np.sqrt(reachable_cstar(c_star, self._epsilon))) for c_star
                            in self._qaoa_data["CONTINUOUS_SOLUTION"]]

        self._quantum_circuit.append(qc_init(warmstart_thetas), range(num_qubits))

        for i in range(p):
            self._quantum_circuit.append(u_problem(gammas[i], num_qubits, bqm), range(num_qubits))
            self._quantum_circuit.append(u_mixer(betas[i], warmstart_thetas), range(num_qubits))

        self._quantum_circuit.measure_all()

    def preprocessing_compatible(self):
        return self._qaoa_data is not None and self._qaoa_data["CONTINUOUS_SOLUTION"] is not None

    def get_name(self):
        return "WARMSTART_CIRCUIT_BUILDER_EPSILON_" + str(self._epsilon)


def qc_init(thetas):
    qc = QuantumCircuit(len(thetas))
    for idx, theta in enumerate(thetas):
        qc.ry(theta, idx)
    return qc


def u_mixer(beta, thetas):
    ws_mixer = QuantumCircuit(len(thetas))
    for idx, theta in enumerate(thetas):
        ws_mixer.ry(-theta, idx)
        ws_mixer.rz(-2 * beta, idx)
        ws_mixer.ry(theta, idx)
    return ws_mixer


def reachable_cstar(c_star: float, epsilon: float) -> float:
    return max(min(c_star, 1 - c_star), epsilon)