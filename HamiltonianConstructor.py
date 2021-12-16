from abc import ABCMeta, abstractmethod
from data import SchedulingData, JobShopSchedulingData
import time

class HamiltonianConstructor(metaclass=ABCMeta):
    def __init__(self):
        self._data = None
        self._hamiltonian = None
        self._T = None
        self._time = 0

    def get_hamiltonian(self, scheduling_data: JobShopSchedulingData, time_span: int, order_bias: float = 1,
                              machine_bias: float = 1, single_run_bias: float = 1, objective_bias: float = 1,
                              variable_pruning: bool = False):
        if self._data is None:
            assert scheduling_data is not None and time_span is not None
            start = time.time()
            self.construct_hamiltonian(scheduling_data, time_span, order_bias, machine_bias, single_run_bias,
                                       objective_bias, variable_pruning)
            end = time.time()
            self._time = end - start
        return self._hamiltonian

    @abstractmethod
    def construct_hamiltonian(self, scheduling_data: JobShopSchedulingData, time_span: int, order_bias: float = 1,
                              machine_bias: float = 1, single_run_bias: float = 1, objective_bias: float = 1,
                              qubit_impugning: bool = False):
        pass

    @abstractmethod
    def get_plottable_solution(self, solution_sampleset, energy_rank=0):
        pass

    @abstractmethod
    def plottable_solution_to_pruned(self, plottable_solution):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def get_T(self):
        return self._T

    def get_time(self):
        return self._time

from HamiltonianConstructors.JobShopHamiltonianConstructor import JobShopHamiltonianConstructor