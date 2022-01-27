from QuantumSchedulers.QAOA.QAOA import ThetaOptimizer, QCSampler, CircuitBuilder
from QuantumSchedulers.QAOA.QCSamplers.LazySampler import LazySampler
from QuantumSchedulers.QAOA.ThetaOptimizers.QiskitMinimizer import expected_value
from qiskit import Aer, BasicAer
from qiskit.algorithms import QAOA
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import COBYLA
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizer,
    SolutionSample,
    OptimizationResultStatus,
)
import numpy as np


class QiskitQAOAOptimizer(ThetaOptimizer):
    def __init__(self, simulator='qasm_simulator'):
        super().__init__()
        self._simulator = simulator
        self._seed = None

    def optimize_theta(self, circuit_builder: CircuitBuilder, qc_sampler: LazySampler, num_reads: int, hamiltonian,
                       theta):
        self._circuit_builder = circuit_builder
        self._qc_sampler = qc_sampler
        self._num_reads = num_reads
        self._hamiltonian = hamiltonian
        self._seed = self._qc_sampler.get_seed()
        optimizer = COBYLA()

        qubo = get_quadradic_program(0.5 * (self._hamiltonian + np.transpose(self._hamiltonian)))
        quantum_instance = QuantumInstance(Aer.get_backend(self._simulator),
                                           seed_simulator=self._seed)
        self._qaoa_mes = QAOA(quantum_instance=quantum_instance, initial_point=theta, reps=int(len(theta)/2),
                              optimizer=optimizer)

        qaoa = MinimumEigenOptimizer(self._qaoa_mes)
        self._solution = qaoa.solve(qubo)
        self._theta = self._solution.min_eigen_solver_result.optimal_point
        counts = self._solution.min_eigen_solver_result.eigenstate
        sum = 0
        counts2 = {}
        for key, prob in counts.items():
            counts[key] = self._num_reads * prob**2
            sum += counts[key]
            counts2[key[::-1]] = counts[key]
        self._qc_sampler.set_counts(counts)
        #assert int(sum) == self._num_reads, "Probabilities do not match"
        self._expected_energy = expected_value(counts, self._num_reads, self._hamiltonian)

    def get_name(self):
        return "QISKITQAOAOPTIMIZER_" + "COBYLA"


def get_quadradic_program(hamiltonian):
    mdl = Model()
    n_qubits = hamiltonian.shape[0]
    x = [mdl.binary_var() for i in range(n_qubits)]
    objective = mdl.sum([hamiltonian[i, i]*x[i] for i in range(n_qubits)])
    objective += mdl.sum([hamiltonian[i, j]*x[i]*x[j] for j in range(n_qubits) for i in range(j)])
    mdl.minimize(objective)
    qp = from_docplex_mp(mdl)
    return qp