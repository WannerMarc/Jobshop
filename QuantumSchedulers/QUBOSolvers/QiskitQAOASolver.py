from QuantumSchedulers.QUBOSolvers.QUBOScheduler import SimpleQUBOSolver, JobShopSchedulingData, HamiltonianConstructor
import numpy as np
from qiskit import Aer, BasicAer
from qiskit.algorithms import QAOA
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils.algorithm_globals import algorithm_globals
from docplex.mp.model import Model
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.translators import from_docplex_mp
import dimod


class QiskitQAOASolver(SimpleQUBOSolver):
    def __init__(self, scheduling_data: JobShopSchedulingData, hamiltonian_constructor: HamiltonianConstructor,
                 time_span: int, p: int, theta=None, optimizer=COBYLA(), simulator='qasm_simulator', order_bias: float = 1,
                 machine_bias: float = 1, single_run_bias: float = 1, objective_bias: float = 1,
                 variable_pruning: bool = False, seed_simulator=937162211):

        super().__init__(scheduling_data, hamiltonian_constructor, time_span, order_bias, machine_bias,
                         single_run_bias, objective_bias, variable_pruning)
        self._optimizer = optimizer
        self._simulator = simulator
        self._theta = theta
        self._p = p
        self._solution = None
        self._seed = seed_simulator
        self._qaoa_mes = None

    def solve(self):
        qubo = self.get_quadradic_program()
        print(qubo)
        quantum_instance = QuantumInstance(Aer.get_backend(self._simulator),
                                           seed_simulator=self._seed)
        self._qaoa_mes = QAOA(quantum_instance=quantum_instance, initial_point=self._theta, reps=self._p)
        qaoa = MinimumEigenOptimizer(self._qaoa_mes)
        self._solution = qaoa.solve(qubo)
        print(self._solution)
        self._sampleset = dimod.SampleSet.from_samples([self._solution.x], 'BINARY', [self._solution.fval],
                                                       num_occurrences=[1])
        print(self._sampleset)

    def get_solver_name(self):
        return "Qiskit QAOA Solver"

    def get_quadradic_program(self):
        mdl = Model()
        n_qubits = self._hamiltonian.shape[0]
        x = [mdl.binary_var() for i in range(n_qubits)]
        objective = mdl.sum([self._hamiltonian[i, i]*x[i] for i in range(n_qubits)])
        objective += mdl.sum([self._hamiltonian[i, j]*x[i]*x[j] for j in range(n_qubits) for i in range(j)])
        mdl.minimize(objective)
        qp = from_docplex_mp(mdl)
        print(n_qubits)
        return qp
        #use from_docplex_mp
        #also add nondiagonal entries

    def draw_quantum_circuit(self):
        self._qaoa_mes.get_optimal_circuit().draw(output='mpl')