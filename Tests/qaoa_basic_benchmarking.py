from QuantumScheduler import QAOASolver, JobShopHamiltonianConstructor
from QuantumSchedulers.QAOA.QAOA import Preprocessor, CircuitBuilder, QCSampler, ThetaOptimizer, Postprocessor
from QuantumSchedulers.QAOA.QAOASolver import QiskitSimulator, QiskitMinimizer, QuboCircuitBuilder
from Reader import JobShopReader
from Scheduler import CPLEXSolver
from typing import List


#if compare_to_optimal is true, T --> Tmin + T, otw T --> T
def test_js_qaoa(problem_filename: str, solution_filenames: List[str], circuit_depths: List[int], compare_to_optimal=True, T=0,
                 num_reads_qaoa=1024, num_reads_eval=5000, theta_init=None, preprocessor: Preprocessor = None,
                 circuit_builder: CircuitBuilder = QuboCircuitBuilder(), qc_sampler: QCSampler = QiskitSimulator(),
                 theta_optimizer: ThetaOptimizer = QiskitMinimizer('COBYLA'), postprocessor: Postprocessor = None,
                 order_bias: float = 1, machine_bias: float = 1, single_run_bias: float = 1, objective_bias: float = 1,
                 variable_pruning: bool = True):
    assert len(solution_filenames) == len(circuit_depths), "Number of solution Filenames and circuit depths not matching"
    reader = JobShopReader()
    reader.read_problem_data(problem_filename)
    data = reader.get_data()

    Tmin = 0
    optimal_plottable_solution = None
    if compare_to_optimal:
        cplex_solver = CPLEXSolver(data)
        cplex_solver.solve()
        optimal_plottable_solution = cplex_solver.get_plottable_solution()
        Tmin = cplex_solver.get_Tmin()

    for i in range(len(circuit_depths)):
        solver = QAOASolver(data, JobShopHamiltonianConstructor(), T+Tmin, circuit_depths[i], theta_init, preprocessor,
                            circuit_builder, qc_sampler, theta_optimizer, postprocessor, order_bias, machine_bias,
                            single_run_bias, objective_bias, variable_pruning)
        solver.solve(num_reads=num_reads_qaoa, optimal_plottable_solution=optimal_plottable_solution,
                     num_reads_eval=num_reads_eval)
        solver.store_soution(solution_filenames[i])
    return
