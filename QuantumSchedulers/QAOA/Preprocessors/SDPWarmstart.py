from QuantumSchedulers.QAOA.QAOA import Preprocessor
import numpy as np
import cvxpy as cp
import copy
from qiskit_optimization.algorithms import CplexOptimizer
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.problems.variable import VarType


class SDPWarmstart(Preprocessor):
    def __init__(self):
        super().__init__()
        self._qaoa_data_name = "CONTINUOUS_SOLUTION"
        self._qaoa_data = {"CONTINUOUS_SOLUTION": None}

    def preprocess(self, hamiltonian=None, scheduling_data=None):
        if hamiltonian is not None:
            self._hamiltonian = hamiltonian
        if scheduling_data is not None:
            self._scheduling_data = scheduling_data

        n = self._hamiltonian.shape[0]
        Q = to_symmetric(self._hamiltonian)
        c = copy.deepcopy(np.diag(Q))
        Q -= np.diag(c)
        X = cp.Variable((n+1, n+1), symmetric=True)
        r = cp.Variable()
        u = cp.Variable(n)
        constraints = [X >> 0]
        constraints += [X[0, 0] == -r]
        for i in range(1, n+1):
            constraints += [X[0, i] == 0.5 * (c[i - 1] + u[i-1])]
            constraints += [X[i, 0] == 0.5 * (c[i - 1] + u[i-1])]
            constraints += [X[i, i] == -u[i-1]]
            for j in range(1, n+1):
                if i != j:
                    constraints += [X[i, j] == Q[i-1, j-1]]

        problem = cp.Problem(cp.Maximize(r), constraints)
        problem.solve()
        qp = get_perturbed_qp(Q, c, u.value)
        for var in qp.variables:
            var.vartype = VarType.CONTINUOUS

        sol = CplexOptimizer().solve(qp)
        self._qaoa_data[self._qaoa_data_name] = sol.x.tolist()
        print(sol)

    def get_name(self):
        return "SDP_WARMSTART"


def to_symmetric(hamiltonian: np.ndarray):
    return 0.5*(hamiltonian + np.transpose(hamiltonian))

def get_perturbed_qp(Q: np.ndarray, c: np.ndarray, u: np.ndarray):
    mdl = Model()
    n_qubits = Q.shape[0]
    x = [mdl.binary_var() for i in range(n_qubits)]
    objective = mdl.sum([(c[i]) * x[i] for i in range(n_qubits)])
    objective += mdl.sum([(u[i]) * x[i] for i in range(n_qubits)])
    objective += mdl.sum([Q[i, j] * x[i] * x[j] for j in range(n_qubits) for i in range(n_qubits)])
    objective -= mdl.sum([u[i] * x[i] * x[i] for i in range(n_qubits)])
    mdl.minimize(objective)
    qp = from_docplex_mp(mdl)
    return qp
