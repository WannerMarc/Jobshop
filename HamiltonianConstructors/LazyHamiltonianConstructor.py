from HamiltonianConstructors.JobShopHamiltonianConstructor import JobShopHamiltonianConstructor, get_label
from data import JobShopSchedulingData
import numpy as np
from pyqubo import Array


class LazyHamiltonianConstructor(JobShopHamiltonianConstructor):
    def __init__(self, hamiltonain):
        super().__init__()
        self._hamiltonian = hamiltonain

    def construct_hamiltonian(self, scheduling_data: JobShopSchedulingData, time_span: int, order_bias: float = 1,
                              machine_bias: float = 1, single_run_bias: float = 1, objective_bias: float = 1,
                              variable_pruning: bool = False):
        self._data = scheduling_data
        self._J, self._m = self._data.get_M.shape
        self._T = time_span
        self._X = Array.create("X_", shape=(self._m * self._J, self._T), vartype="BINARY")
        self._M = self._data.get_M
        self._P = self._data.get_P
        self._labels = [get_label(idx, t) for idx, t in np.ndindex(self._m * self._J, self._T)]
        print(self._hamiltonian)

    def get_name(self):
        return "LAZY_HAMILTONIAN_CONSTRUCTOR"

    def get_plottable_solution(self, solution_sampleset, energy_rank=0):
        plottable_solution = np.zeros((self._J, self._m, self._T), dtype=np.int)
        return plottable_solution