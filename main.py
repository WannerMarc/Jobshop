from Reader import JobShopReader
from HamiltonianConstructor import JobShopHamiltonianConstructor
from QuantumScheduler import SASolver, QASolver, QAOASolver, QiskitQAOASolver, CPLEXWarmstart, SDPWarmstart, \
    SimpleWarmstart, WarmstartCircuitBuilder, IBMRealDevice
from Scheduler import CPLEXSolver, ResultPlotter
import matplotlib.pyplot as plt
from qiskit import Aer, transpile
from QuantumSchedulers.QAOA.QAOASolver import expected_value
import numpy as np
from JobShopSampler import JobShopSampler, nnull_condition
import os
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from Tests.qaoa_basic_benchmarking import test_js_qaoa, compare_qaoa_versions, plot_result_times, plot_result_times_single_iteration
from qiskit import IBMQ
"""Construction ongoing, main file is only used for trying out code"""

def test_sampler():
    this_path = os.path.dirname(__file__) + os.sep + "Problems"
    a = 1
    b = None
    print(nnull_condition(a, b))
    jssampler = JobShopSampler(this_path, 3, 3, 1, 3)
    jssampler.sample(nsamples=10)

def test_qaoa():
    reader = JobShopReader()
    reader.read_problem_data("Problems/micro_example.txt")
    data = reader.get_data()

    solvers = []
    #solvers.append(CPLEXSolver(data))
    #solvers.append(SASolver(data, JobShopHamiltonianConstructor(), 6, variable_pruning=True))
    #solvers.append(QASolver(data, JobShopHamiltonianConstructor(), 6, variable_pruning=True))
    hc = JobShopHamiltonianConstructor()
    for i in range(20):
        solvers.append(QAOASolver(data, hc, 6, 2, variable_pruning=True))
    #solvers[0].solve()
    # plottable_solution = solvers[0].get_plottable_solution()
    #solvers[0].plot_solution()
    for solver in solvers:
        solver.solve()
        solver.plot_solution()
        # solver.plot_success_probability_heatmap((100, 100), plottable_solution, 10000)

    plt.show()

def theta_test(qaoa_solver):
    qaoa_solver._circuit_builder.set_bqm(qaoa_solver._hamiltonian, qaoa_solver._num_qubits)
    qaoa_solver._circuit_builder.set_bqm(qaoa_solver._hamiltonian - np.identity(qaoa_solver._num_qubits),
    qaoa_solver._num_qubits)
    thetas = [[0.5, 1], [1, 1], [2, 2], [2, 1], [2, 5]]
    for theta in thetas:
        qc = qaoa_solver._circuit_builder.get_quantum_circuit(theta=theta)
        qc.decompose().decompose().draw(output='mpl')
        plt.show()
        print(qaoa_solver._qc_sampler.sample_qc(qc, 512))
        backend = Aer.get_backend('aer_simulator')
        qobj = transpile(qc, backend)
        counts = backend.run(qobj, validate=True, seed_simulator=7, shots=512).result().get_counts(qc)
        print(expected_value(counts, 512, qaoa_solver._hamiltonian))
        # qaoa_solver._circuit_builder.set_bqm(qaoa_solver._hamiltonian - np.identity(qaoa_solver._num_qubits),
        # qaoa_solver._num_qubits)

def compare_qaoas(data):
    qiskit_qaoa = QiskitQAOASolver(data, JobShopHamiltonianConstructor(), 4, 1, variable_pruning=True,
                                       objective_bias=0,
                                       theta=[1, 1])
    qaoa = QAOASolver(data, JobShopHamiltonianConstructor(), 4, 1, variable_pruning=True, objective_bias=0,
                          theta=[1, 1])
    qiskit_qaoa.solve()
    # qaoa.solve()
    qiskit_qaoa.plot_solution()
    # qaoa.plot_solution()
    qiskit_qaoa.draw_quantum_circuit()
    qaoa.draw_quantum_circuit()
    plt.show()


def main():

    problem = "Problems/nano_example.txt"
    solution = "micro_example.txt"
    ps = [1, 2]
    solutions = ["Solutions" + os.sep + "res" + str(p) + "_" + solution for p in ps]
    reader = JobShopReader()
    reader.read_problem_data(problem)
    data = reader.get_data()
    solver = CPLEXSolver(data)
    solver.solve()
    solver.plot_solution()
    plt.show()
    hamiltonian_constructor = JobShopHamiltonianConstructor()
    preprocessor = SDPWarmstart()
    preprocessor2 = SimpleWarmstart()
    solver = QAOASolver(data, hamiltonian_constructor, 4, 1, variable_pruning=True, objective_bias=0,
                        qc_sampler=IBMRealDevice(IBMQ.load_account(), 'ibmq_lima'))
    solver8 = QAOASolver(data, hamiltonian_constructor, 4, 1, variable_pruning=True, objective_bias=0)
    compare_qaoa_versions("Solutions_real_device"+os.sep, [solver, solver8], range(1, 3), num_samples_per_setup=5,
                          num_reads_opt=1000, num_reads_eval=10000, solver_names=["Real_device_QAOA", "QAOA"], mode='PLOT_ONLY')
    plt.show()
    """
    solver = CPLEXSolver(data)
    solver.solve()
    solver.plot_solution()
    plt.show()
    """
    solver1 = QAOASolver(data, hamiltonian_constructor, 4, 3, variable_pruning=True, objective_bias=1)

    solver2 = QAOASolver(data, hamiltonian_constructor, 4, 3, variable_pruning=True, objective_bias=1,
                         preprocessor=preprocessor, circuit_builder=WarmstartCircuitBuilder(epsilon=0.25))
    solver3 = QAOASolver(data, hamiltonian_constructor, 4, 3, variable_pruning=True, objective_bias=1,
                         preprocessor=preprocessor, circuit_builder=WarmstartCircuitBuilder(epsilon=0))
    solver4 = QAOASolver(data, hamiltonian_constructor, 4, 3, variable_pruning=True, objective_bias=1,
                         preprocessor=preprocessor2, circuit_builder=WarmstartCircuitBuilder(epsilon=0.25))
    solver5 = QAOASolver(data, hamiltonian_constructor, 4, 3, variable_pruning=True, objective_bias=0,
                         preprocessor=preprocessor, circuit_builder=WarmstartCircuitBuilder(epsilon=0))
    solver6 = QAOASolver(data, hamiltonian_constructor, 4, 3, variable_pruning=True, objective_bias=0,
                         preprocessor=preprocessor, circuit_builder=WarmstartCircuitBuilder(epsilon=0.4))
    solver7 = QAOASolver(data, hamiltonian_constructor, 4, 3, variable_pruning=True, objective_bias=0,
                         preprocessor=preprocessor, circuit_builder=WarmstartCircuitBuilder(epsilon=0.5))


    compare_qaoa_versions("Solutions_nondegenerate" + os.sep, [solver1, solver2, solver3, solver4],
                          range(1, 6),
                          num_samples_per_setup=100, num_reads_opt=1000,
                          num_reads_eval=10000, solver_names=["QAOA", "opt-WS-QAOA-0.25", "opt-WS-QAOA-0", "simple-WS-QAOA-0.25"])

    #plot_result_times("Solutions_maxp" + os.sep)

    #plotter = ResultPlotter("Solutions\\res_0_1_0.txt")
    #plotter.plot_solution()
    plt.show()
    '''test_js_qaoa(problem, solutions, ps)
    plotter = ResultPlotter(solutions[0])
    plotter.plot_solution()
    plt.show()
    '''
    #test_js_qaoa(problem, solutions, ps)

    """
    qaoa_solver._circuit_builder.set_bqm(qaoa_solver._hamiltonian, qaoa_solver._num_qubits)
    #qaoa_solver._circuit_builder.set_bqm(qaoa_solver._hamiltonian - np.identity(qaoa_solver._num_qubits),
                                         #qaoa_solver._num_qubits)
    thetas = [[0.5, 1], [1, 1], [2, 2], [2, 1], [2, 5]]
    for theta in thetas:

        qc = qaoa_solver._circuit_builder.get_quantum_circuit(theta=theta)
        qc.decompose().decompose().draw(output='mpl')
        plt.show()
        print(qaoa_solver._qc_sampler.sample_qc(qc, 512))
        backend = Aer.get_backend('aer_simulator')
        qobj = transpile(qc, backend)
        counts = backend.run(qobj, validate=True, seed_simulator=7, shots=512).result().get_counts(qc)
        print(expected_value(counts, 512, qaoa_solver._hamiltonian))
        #qaoa_solver._circuit_builder.set_bqm(qaoa_solver._hamiltonian - np.identity(qaoa_solver._num_qubits),
                                             #qaoa_solver._num_qubits)
    """
    #qaoa_solver.plot_expectation_heatmap((25, 50), 1000)
    #qaoa_solver.solve()
    #qaoa_solver.plot_solution()
    #plt.show()



if __name__ == "__main__":
    main()