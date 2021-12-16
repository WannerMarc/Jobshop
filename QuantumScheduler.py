from abc import ABCMeta, abstractmethod
import sys
sys.path.append('..')
from HamiltonianConstructor import HamiltonianConstructor, JobShopHamiltonianConstructor
from data import JobShopSchedulingData
from Reader import JobShopReader
from Scheduler import JobShopScheduler
from dimod import BinaryQuadraticModel


class QCJobShopScheduler(JobShopScheduler, metaclass=ABCMeta):
    def __init__(self, scheduling_data: JobShopSchedulingData, hamiltonian_constructor: HamiltonianConstructor,
                 time_span: int, order_bias: float = 1, machine_bias: float = 1, single_run_bias: float = 1,
                 objective_bias: float = 1, variable_pruning: bool = False):
        self._data = scheduling_data
        self._plottable_solution = None
        self._benchmarking_data = {}
        self._sampleset = None
        self._hamiltonian_constructor = hamiltonian_constructor
        self._hamiltonian = self._hamiltonian_constructor.get_hamiltonian(scheduling_data, time_span, order_bias,
                                                                    machine_bias, single_run_bias, objective_bias,
                                                                    variable_pruning)
        self._bqm = BinaryQuadraticModel.from_qubo(self._hamiltonian)
        self._num_qubits = self._hamiltonian.shape[0]
    @abstractmethod
    def solve(self):
        pass

    def get_plottable_solution(self, energy_rank: int = 0):
        assert self._sampleset is not None
        self._plottable_solution = self._hamiltonian_constructor.get_plottable_solution(self._sampleset, energy_rank)

from QuantumSchedulers.QUBOSolvers.QUBOScheduler import SASolver, QASolver, SimpleQUBOSolver, QiskitQAOASolver
from QuantumSchedulers.QAOA.QAOASolver import QAOASolver
from QuantumSchedulers.QAOA.Preprocessors.CPLEXWarmstart import CPLEXWarmstart
from QuantumSchedulers.QAOA.Preprocessors.SDPWarmstart import SDPWarmstart
from QuantumSchedulers.QAOA.Preprocessors.SimpleWarmstart import SimpleWarmstart
from QuantumSchedulers.QAOA.CircuitBuilders.WarmstartCircuitBuilder import WarmstartCircuitBuilder
from QuantumSchedulers.QAOA.QCSamplers.IBMRealDevice import IBMRealDevice