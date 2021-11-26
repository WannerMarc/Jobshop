from QuantumSchedulers.QAOA.QAOA import Preprocessor
import numpy as np
from QuantumSchedulers.QAOA.Preprocessors.SDPWarmstart import to_symmetric
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.problems.variable import VarType
from qiskit_optimization.algorithms import CplexOptimizer
import copy


class SimpleWarmstart(Preprocessor):
    def __init__(self):
        super().__init__()
        self._qaoa_data_name = "CONTINUOUS_SOLUTION"
        self._qaoa_data = {"CONTINUOUS_SOLUTION": None}

    def preprocess(self, hamiltonian=None, scheduling_data=None):
        if hamiltonian is not None:
            self._hamiltonian = hamiltonian
        if scheduling_data is not None:
            self._scheduling_data = scheduling_data

        Q = to_symmetric(self._hamiltonian)
        c = copy.deepcopy(np.diag(Q))
        Q -= np.diag(c)

        qp = get_simple_convex_qp(Q, c)
        for var in qp.variables:
            var.vartype = VarType.CONTINUOUS

        sol = CplexOptimizer().solve(qp)
        print(np.dot(np.dot(sol.x, Q), sol.x)+np.dot(sol.x, c))
        print(np.dot(np.dot(sol.x, hamiltonian), sol.x))
        self._qaoa_data[self._qaoa_data_name] = sol.x.tolist()
        print(sol)


    def get_name(self):
        return "SIMPLE_WARMSTART"


def get_simple_convex_qp(Q, c):
    mdl = Model()
    n_qubits = Q.shape[0]
    x = [mdl.binary_var() for i in range(n_qubits)]
    eigvals = np.linalg.eigvalsh(Q)
    u = eigvals[0]*np.ones(n_qubits)
    objective = mdl.sum([(c[i]) * x[i] for i in range(n_qubits)])
    objective += mdl.sum([(u[i]) * x[i] for i in range(n_qubits)])
    objective += mdl.sum([Q[i, j] * x[i] * x[j] for j in range(n_qubits) for i in range(n_qubits)])
    objective -= mdl.sum([u[i] * x[i] * x[i] for i in range(n_qubits)])
    mdl.minimize(objective)
    qp = from_docplex_mp(mdl)
    return qp