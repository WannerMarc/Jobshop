from abc import ABCMeta, abstractmethod
import sys
sys.path.append('../..')
from QuantumScheduler import QCJobShopScheduler, HamiltonianConstructor, JobShopSchedulingData


class SimpleQUBOSolver(QCJobShopScheduler, metaclass=ABCMeta):
    def __init__(self, scheduling_data: JobShopSchedulingData, hamiltonian_constructor: HamiltonianConstructor,
                 time_span: int, order_bias: float = 1, machine_bias: float = 1, single_run_bias: float = 1,
                 objective_bias: float = 1, variable_pruning: bool = False):
        super().__init__(scheduling_data, hamiltonian_constructor, time_span, order_bias,
                                                                    machine_bias, single_run_bias, objective_bias,
                                                                    variable_pruning)

    @abstractmethod
    def solve(self):
        pass


from QuantumSchedulers.QUBOSolvers.SASolver import SASolver
from QuantumSchedulers.QUBOSolvers.QASolver import QASolver
from QuantumSchedulers.QUBOSolvers.QiskitQAOASolver import QiskitQAOASolver