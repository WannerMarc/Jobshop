from QuantumSchedulers.QUBOSolvers.QUBOScheduler import SimpleQUBOSolver, JobShopSchedulingData, HamiltonianConstructor
from dwave.system import EmbeddingComposite, DWaveSampler


class QASolver(SimpleQUBOSolver):
    def __init__(self, scheduling_data: JobShopSchedulingData, hamiltonian_constructor: HamiltonianConstructor,
                 time_span: int, order_bias: float = 1, machine_bias: float = 1, single_run_bias: float = 1,
                 objective_bias: float = 1, variable_pruning: bool = False):
        super().__init__(scheduling_data, hamiltonian_constructor, time_span, order_bias, machine_bias,
                                       single_run_bias, objective_bias, variable_pruning)

    def solve(self, num_reads: int = 100, energy_rank: int = 0):
        sampler = EmbeddingComposite(DWaveSampler())
        self._sampleset = sampler.sample_qubo(self._hamiltonian, num_reads=num_reads)

    def get_solver_name(self):
        return "Quantum Annealing Solver"