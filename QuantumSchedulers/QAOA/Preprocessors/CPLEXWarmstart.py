from QuantumSchedulers.QAOA.QAOA import Preprocessor
from QuantumSchedulers.QUBOSolvers.QiskitQAOASolver import get_quadradic_program
from qiskit_optimization.problems.variable import VarType
from qiskit_optimization.algorithms import CplexOptimizer


class CPLEXWarmstart(Preprocessor):
    def __init__(self):
        super().__init__()
        self._qaoa_data_name = "CONTINUOUS_SOLUTION"
        self._qaoa_data = {"CONTINUOUS_SOLUTION": None}

    def preprocess(self, hamiltonian=None, scheduling_data=None):
        if hamiltonian is not None:
            self._hamiltonian = hamiltonian
        if scheduling_data is not None:
            self._scheduling_data = scheduling_data

        qp = get_quadradic_program(self._hamiltonian)
        for var in qp.variables:
            var.vartype = VarType.CONTINUOUS

        sol = CplexOptimizer().solve(qp)
        self._qaoa_data[self._qaoa_data_name] = sol.x

    def get_name(self):
        return "CPLEX_WARMSTART"