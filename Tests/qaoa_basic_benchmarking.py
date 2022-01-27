from QuantumScheduler import QAOASolver, JobShopHamiltonianConstructor
from QuantumSchedulers.QAOA.QAOA import Preprocessor, CircuitBuilder, QCSampler, ThetaOptimizer, Postprocessor
from QuantumSchedulers.QAOA.QAOASolver import QiskitSimulator, QiskitMinimizer, QuboCircuitBuilder
from Reader import JobShopReader
from Scheduler import CPLEXSolver
from typing import List
import os
import pandas as pd
import numpy as np
import json
import seaborn as sns


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


def compare_qaoa_versions(solution_directory, solvers: List[QAOASolver], p_range: List[int],
                          num_samples_per_setup=10, num_reads_opt=1000, num_reads_eval=10000,
                          solver_names: List[str]=None, mode='COMPUTE_AND_PLOT', title="QAOA accuracy", plot='line'):
    # fill in solver names
    _solver_names = solver_names
    if solver_names is None:
        _solver_names = []
    if len(_solver_names) < len(solvers):
        for i in range(len(solvers) - len(_solver_names)):
            _solver_names.append(solvers[len(solvers) - len(_solver_names) + i].get_solver_name())
    # solver names need to be unique for seaborn lineplot
    _solver_names = make_names_unique(_solver_names)

    def sol_idx_to_filename(solver_idx: int, p: int, sample_idx: int):
        return "res_" + _solver_names[solver_idx] + "_" + str(p) + "_" + str(sample_idx) + ".txt"

    def filename_to_sol_idx(filename: str):
        name = filename.split('.')[0]
        indices = name.split('_')
        return [indices[k] for k in range(1, 4)]

    if mode == 'COMPUTE_AND_PLOT' or mode == 'COMPUTE_ONLY':
        # find optimal solution classically as a benchmark
        cplex_solver = CPLEXSolver(solvers[0].get_data())
        cplex_solver.solve()
        optimal_plottable_solution = cplex_solver.get_plottable_solution()
        solver_idx = 0
        for solver in solvers:
            for p in p_range:
                solver.set_p(p)
                for i in range(num_samples_per_setup):
                    json_filename = sol_idx_to_filename(solver_idx, p, i)
                    solver.reset_theta()
                    solver.solve(num_reads=num_reads_opt, num_reads_eval=num_reads_eval,
                                 optimal_plottable_solution=optimal_plottable_solution)
                    solver.store_solution(solution_directory + os.sep + json_filename)
            solver_idx += 1

    if mode == 'COMPUTE_AND_PLOT' or mode == 'PLOT_ONLY':
        data = []
        for solver_idx in range(len(_solver_names)):
            for p in p_range:
                for i in range(num_samples_per_setup):
                    filename = solution_directory + os.sep + sol_idx_to_filename(solver_idx, p, i)
                    with open(filename) as json_file:
                        result_data = json.load(json_file)
                        success_probability = result_data["SUCCESS_PROBABILITY"]
                        if plot == 'scatter':
                            p_new = p + solver_idx/(2 * len(solvers) - 2) - 0.25
                            data.append([p_new, success_probability, _solver_names[solver_idx]])
                        else:
                            data.append([p, success_probability, _solver_names[solver_idx]])

        df = pd.DataFrame(np.array(data), columns=['p', 'success_probability', 'solver_name'])
        df = df.astype({'success_probability': 'float64', 'p': 'int32'})
        markers = ["v" for solver in solvers]
        if plot == 'line':
            sns.lineplot(data=df, x="p", y='success_probability', hue='solver_name', style='solver_name', dashes=False,
                        markers=markers).set_title(title)
        if plot == 'scatter':
            sns.scatterplot(data=df, x="p", y='success_probability', hue='solver_name', style='solver_name',
                        markers=markers, estimator=None).set_title(title)


def make_names_unique(names: List[str]):
    unique_names = []
    for name in names:
        if name in unique_names:
            i = 1
            while name + str(i) in unique_names:
                i += 1
            unique_names.append(name + str(i))
        else:
            unique_names.append(name)
    return unique_names


#PRE: Folder with only solution files
def plot_result_times(solution_directory):
    data = []
    for filename in os.listdir(solution_directory):
        with open(solution_directory + filename) as json_file:
            result_data = json.load(json_file)
            p = result_data["P"]
            time = result_data["TIME"]
            circuit_building_time = (time["REPS"] + 1)*time["CIRCUIT_BUILDER"]
            qc_sampling_time = (time["REPS"] + 1) * time["QCSAMPLER"]
            optimizing_time = time["THETAOPTIMIZER"]
            data.append([p, circuit_building_time, "CIRCUIT_BUILDER"])
            data.append([p, qc_sampling_time, "QCSAMPLER"])
            data.append([p, optimizing_time, "THETAOPTIMIZER"])
    df = pd.DataFrame(np.array(data), columns=['p', 'runtime(s)', 'QAOA_parts'])
    df = df.astype({'runtime(s)': 'float64', 'p': 'int32'})
    markers = ["v" for i in range(3)]
    sns.lineplot(data=df, x="p", y='runtime(s)', hue='QAOA_parts', style='QAOA_parts', dashes=False,
                 markers=markers).set_title("QAOA simulator runtimes")


def plot_result_times_single_iteration(solution_directory):
    data = []
    for filename in os.listdir(solution_directory):
        with open(solution_directory + filename) as json_file:
            result_data = json.load(json_file)
            p = result_data["P"]
            time = result_data["TIME"]
            circuit_building_time = time["CIRCUIT_BUILDER"]
            qc_sampling_time = time["QCSAMPLER"]
            data.append([p, circuit_building_time, "CIRCUIT_BUILDER"])
            data.append([p, qc_sampling_time, "QCSAMPLER"])
    df = pd.DataFrame(np.array(data), columns=['p', 'runtime(s)', 'QAOA_parts'])
    df = df.astype({'runtime(s)': 'float64', 'p': 'int32'})
    markers = ["v" for i in range(2)]
    sns.lineplot(data=df, x="p", y='runtime(s)', hue='QAOA_parts', style='QAOA_parts', dashes=False,
                 markers=markers).set_title("Runtimes for Single Optimization step")


