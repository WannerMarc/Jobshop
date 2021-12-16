from QuantumSchedulers.QAOA.CircuitBuilders.QuboCircuitBuilder import QuboCircuitBuilder
from QuantumSchedulers.QAOA.QCSamplers.QiskitSimulator import QiskitSimulator
from QuantumSchedulers.QAOA.ThetaOptimizers.QiskitMinimizer import QiskitMinimizer, get_expectation, expected_value, \
    key_to_vector
from QuantumScheduler import QCJobShopScheduler, JobShopSchedulingData, HamiltonianConstructor
from QuantumSchedulers.QAOA.QAOA import Preprocessor, CircuitBuilder, QCSampler, ThetaOptimizer, Postprocessor
from random import random
import math
import matplotlib.pyplot as plt
import dimod
import numpy as np
from qiskit.visualization import plot_histogram
import seaborn as sns
from dimod import SampleSet


class QAOASolver(QCJobShopScheduler):
    def __init__(self, scheduling_data: JobShopSchedulingData, hamiltonian_constructor: HamiltonianConstructor,
                 time_span: int, p: int, theta=None, preprocessor: Preprocessor = None,
                 circuit_builder: CircuitBuilder = QuboCircuitBuilder(), qc_sampler: QCSampler = QiskitSimulator(),
                 theta_optimizer: ThetaOptimizer = QiskitMinimizer('COBYLA'), postprocessor: Postprocessor = None,
                 order_bias: float = 1, machine_bias: float = 1, single_run_bias: float = 1, objective_bias: float = 1,
                 variable_pruning: bool = False):
        super().__init__(scheduling_data, hamiltonian_constructor, time_span, order_bias, machine_bias,
                                       single_run_bias, objective_bias, variable_pruning)
        self._preprocessor = preprocessor
        self._circuit_builder = circuit_builder
        self._qc_sampler = qc_sampler
        self._theta_optimizer = theta_optimizer
        self._postprocessor = postprocessor
        self._p = p
        self._theta = theta
        if theta is None:
            self._theta = default_init_theta(p)
        self._qaoa_data = None
        self._quantum_circuit = None
        self._counts = None
        self._total_counts = None
        self._Tmin = None
        self._time: dict = {"HAMILTONIAN_CONSTRUCTOR": "NONE", "PREPROCESSOR": "NONE", "CIRCUIT_BUILDER": "NONE",
                            "QCSAMPLER": "NONE", "THETAOPTIMIZER": "NONE", "POSTPROCESSOR": "NONE", "REPS": "NONE"}
        self.init_benchmarking()

    def solve(self, num_reads=1000, energy_rank=0, optimal_plottable_solution=None, num_reads_eval=10000,
              preprocess_only=False):
        self.reset_times()
        if self._preprocessor is not None:
            self._qaoa_data = self._preprocessor.get_preprocess_data(self._hamiltonian, self._data)
        if preprocess_only:
            return
        # adjust circuit builder so that it only depends on theta
        self._circuit_builder.set_bqm(self._hamiltonian, self._num_qubits)
        self._circuit_builder.set_preprocess_data(self._qaoa_data)
        # run QAOA
        self._theta = self._theta_optimizer.get_theta(self._hamiltonian, self._theta, num_reads, self._circuit_builder,
                                                      self._qc_sampler)
        self._quantum_circuit = self._circuit_builder.get_quantum_circuit(self._theta, self._hamiltonian, self._num_qubits)
        self._counts = self._qc_sampler.get_counts(self._quantum_circuit, num_reads_eval)
        if self._postprocessor is not None:
            postprocessing_input = None  # tbd, postprocessing input is a dummy, replace by arguments like qc, etc when
                                         # known what is needed
            self._counts = self._postprocessor.get_postprocessed_data(postprocessing_input)
        self._sampleset = to_sampleset(self._counts, self._hamiltonian)
        self._total_counts = num_reads
        optimal_plottable_solution = self.extend_optimal_plottable_solution(optimal_plottable_solution)
        self.update_benchmarking(optimal_plottable_solution, num_reads_eval)
        print(self._benchmarking_data)

    def get_solver_name(self):
        return "QAOA"

    def plot_expectation_heatmap(self, shape, num_reads):
        self._circuit_builder.set_bqm(self._hamiltonian, self._num_qubits)
        expectation = get_expectation(self._hamiltonian, self._circuit_builder, self._qc_sampler, num_reads)
        beta_stepsize = math.pi/shape[0]
        gamma_stepsize = 2*math.pi/shape[1]
        result = np.zeros(shape)
        for i, j in np.ndindex(shape):
            result[i, j] = expectation([(shape[0]-i)*beta_stepsize, j*gamma_stepsize])
            print((shape[0]-i) * beta_stepsize, j * gamma_stepsize, result[i, j])
        fig, ax = plt.subplots()
        ax = sns.heatmap(result, center=0, xticklabels=['0', r'$\pi$', r'$2\pi$'], yticklabels=[r'$\pi$', '0'])
        ax.set_title("Energy landscape")
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel(r'$\beta$')
        ax.set_xticks([0, shape[1]/2, shape[1]])
        ax.set_yticks([0, shape[0]])

    def get_success_probability(self, theta, optimal_energy, num_reads, run_simulation=True):
        counts = self._counts
        if run_simulation:
            qc = self._circuit_builder.get_quantum_circuit(theta, self._hamiltonian, self._num_qubits)
            counts = self._qc_sampler.get_counts(qc, num_reads)
        nsolved = 0
        energy_counts = to_energy_counts(counts, self._hamiltonian)
        for energy_count in energy_counts:
            if energy_count[1] <= optimal_energy:
                nsolved += energy_count[2]
            if energy_count[1] < optimal_energy:
                print("Smaller energy than optimal, Difference : ", optimal_energy - energy_count[1], " Value: ",
                      energy_counts[0])

        return nsolved/num_reads

    def plot_success_probability_heatmap(self, shape, optimal_plottable_solution, num_reads: int):
        optimal_plottable_solution = self.extend_optimal_plottable_solution(optimal_plottable_solution)
        optimal_energy = self.get_optimal_energy(optimal_plottable_solution)
        self._circuit_builder.set_bqm(self._hamiltonian, self._num_qubits)
        beta_stepsize = math.pi / shape[0]
        gamma_stepsize = 2 * math.pi / shape[1]
        result = np.zeros(shape)
        for i, j in np.ndindex(shape):
            result[i, j] = self.get_success_probability([(shape[0] - i) * beta_stepsize, j * gamma_stepsize],
                                                        optimal_energy, num_reads)
            print((shape[0] - i) * beta_stepsize, j * gamma_stepsize, result[i, j])
        fig, ax = plt.subplots()
        ax = sns.heatmap(result, center=0, xticklabels=['0', r'$\pi$', r'$2\pi$'], yticklabels=[r'$\pi$', '0'])
        ax.set_title("Probability for finding the right solution")
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel(r'$\beta$')
        ax.set_xticks([0, shape[1] / 2, shape[1]])
        ax.set_yticks([0, shape[0]])

    def get_optimal_energy(self, optimal_plottable_solution):
        optimal_plottable_solution = self.extend_optimal_plottable_solution(optimal_plottable_solution)
        # remove the ones from the plottable solution so that it only starts once
        J, m, T = optimal_plottable_solution.shape
        for i, o in np.ndindex(J, m):
            for t in range(T):
                if optimal_plottable_solution[i, o, t] == 1:
                    optimal_plottable_solution[i, o, t + 1: t + self._data.get_P[i, o]] = \
                        np.zeros((self._data.get_P[i, o] - 1))

        reduced_solution = self._hamiltonian_constructor.plottable_solution_to_pruned(optimal_plottable_solution)
        optimal_energy = np.dot(np.dot(reduced_solution, self._hamiltonian), reduced_solution)
        return optimal_energy

    def extend_optimal_plottable_solution(self, optimal_plottable_solution):
        if optimal_plottable_solution is None:
            return None
        T = self._hamiltonian_constructor.get_T()
        J, m, self._Tmin = optimal_plottable_solution.shape
        assert T >= self._Tmin, "Not enough time steps to find optimal solution"
        assert optimal_plottable_solution is not None, "Optimal plottable solution can not be None"

        return np.append(optimal_plottable_solution, np.zeros((J, m, T-self._Tmin), dtype=np.int), axis=2)

    def init_benchmarking(self):
        #Methods
        self._benchmarking_data["SOLVER_NAME"] = self.get_solver_name()
        self._benchmarking_data["PREPROCESSOR"] = "NONE"
        self._benchmarking_data["POSTPROCESSOR"] = "NONE"
        if self._preprocessor is not None:
            self._benchmarking_data["PREPROCESSOR"] = self._preprocessor.get_name()
        if self._postprocessor is not None:
            self._benchmarking_data["POSTPROCESSOR"] = self._postprocessor.get_name()
        self._benchmarking_data["HAMILTONIAN_CONSTRUCTOR"] = self._hamiltonian_constructor.get_name()
        self._benchmarking_data["CIRCUIT_BUILDER"] = self._circuit_builder.get_name()
        self._benchmarking_data["QCSAMPLER"] = self._qc_sampler.get_name()
        self._benchmarking_data["THETA_OPTIMIZER"] = self._theta_optimizer.get_name()
        #Input data
        self._benchmarking_data["PROBLEM_FILENAME"] = self._data.get_filename
        self._benchmarking_data["NUM_READS"] = "NONE"
        self._benchmarking_data["NUM_READS_EVAL"] = "NONE"
        self._benchmarking_data["SIZE"] = self._data.get_M.shape
        self._benchmarking_data["T"] = self._hamiltonian_constructor.get_T()
        self._benchmarking_data["T_MIN"] = "NONE"
        self._benchmarking_data["NUM_QUBITS"] = self._num_qubits
        self._benchmarking_data["P"] = self._p
        self._benchmarking_data["SEED"] = self._qc_sampler.get_seed()
        self._benchmarking_data["THETA_INIT"] = self._theta
        #Results
        self._benchmarking_data["THETA"] = self._theta
        self._benchmarking_data["EXPECTED_ENERGY"] = "NONE"
        self._benchmarking_data["OPTIMAL_ENERGY"] = "NONE"
        self._benchmarking_data["SUCCESS_PROBABILITY"] = "NONE"
        self._benchmarking_data["SOLUTION_ENERGY"] = "NONE"
        self._benchmarking_data["SOLUTION_PROBABILITY"] = "NONE"
        self._benchmarking_data["QAOA_DATA"] = "NONE"
        self._benchmarking_data["TIME"] = self._time

    def update_benchmarking(self, optimal_plottable_solution, num_reads_eval):
        self.update_times()
        #Input data
        self._benchmarking_data["NUM_READS"] = self._total_counts
        self._benchmarking_data["NUM_READS_EVAL"] = num_reads_eval
        #Results
        self._benchmarking_data["THETA"] = self._theta.tolist()
        self._benchmarking_data["EXPECTED_ENERGY"] = self._theta_optimizer.get_expected_energy()
        if optimal_plottable_solution is not None:
            optimal_energy = self.get_optimal_energy(optimal_plottable_solution)
            self._benchmarking_data["OPTIMAL_ENERGY"] = optimal_energy
            self._benchmarking_data["SUCCESS_PROBABILITY"] = self.get_success_probability(self._theta, optimal_energy,
                                                                                          num_reads_eval, run_simulation=False)
            self._benchmarking_data["T_MIN"] = self._Tmin
        self._benchmarking_data["SOLUTION_ENERGY"] = self._sampleset.first.energy
        self._benchmarking_data["SOLUTION_PROBABILITY"] = self._sampleset.first.num_occurrences/num_reads_eval
        self._benchmarking_data["QAOA_DATA"] = self._qaoa_data
        self._benchmarking_data["TIME"] = self._time

    def draw_quantum_circuit(self):
        if self._quantum_circuit is None:
            self._quantum_circuit = self._circuit_builder.get_quantum_circuit(self._theta, self._hamiltonian, self._num_qubits)
        return self._quantum_circuit.decompose().decompose().draw(output='mpl')

    def set_p(self, p):
        self._p = p
        self._benchmarking_data["P"] = self._p

    def reset_theta(self, theta=None):
        if theta is not None:
            assert len(theta) == 2*self._p, "Set p to " + str(len(theta)/2) + " first before resetting theta"
        else:
            self._theta = default_init_theta(self._p)
            self._benchmarking_data["THETA_INIT"] = self._theta
            self._benchmarking_data["THETA"] = self._theta

    def reset_times(self):
        self._circuit_builder.reset_time()
        self._qc_sampler.reset_time()

    def update_times(self):
        self._time["HAMILTONIAN_CONSTRUCTOR"] = self._hamiltonian_constructor.get_time()
        if self._preprocessor is not None:
            self._time["PREPROCESSOR"] = self._preprocessor.get_time()
        self._time["CIRCUIT_BUILDER"] = self._circuit_builder.get_time()
        self._time["QCSAMPLER"] = self._qc_sampler.get_time()
        self._time["THETAOPTIMIZER"] = self._theta_optimizer.get_time()
        if self._postprocessor is not None:
            self._time["POSTPROCESSOR"] = self._postprocessor.get_time()
        self._time["REPS"] = self._qc_sampler.get_nreps() - 1


def default_init_theta(p):
    return [math.pi * (1 + int(i/p)) * random() for i in range(2*p)]


def key_to_dict(key: str):
    res_dict = {}
    for i in range(len(key)):
        res_dict[i] = int(key[i])
    return res_dict


def to_energy_counts(counts: dict, hamiltonian):
    energy_counts = []
    for key, count in counts.items():
        x = key_to_vector(key)
        energy = np.dot(np.dot(x, hamiltonian), x)
        energy_counts.append((key_to_dict(key), energy, count))

    return energy_counts


def to_sampleset(counts: dict, hamiltonian):
    energy_counts = to_energy_counts(counts, hamiltonian)
    energy_counts.sort(key=lambda x: x[1])
    variables = [energy_count[0] for energy_count in energy_counts]
    energies = [energy_count[1] for energy_count in energy_counts]
    num_ocs = [energy_count[2] for energy_count in energy_counts]

    return dimod.SampleSet.from_samples(variables, 'BINARY', energies, num_occurrences=num_ocs)

