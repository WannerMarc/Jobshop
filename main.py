from Reader import JobShopReader
from HamiltonianConstructor import JobShopHamiltonianConstructor
from QuantumScheduler import SASolver, QASolver, QAOASolver, QiskitQAOASolver
from Scheduler import CPLEXSolver, ResultPlotter
import matplotlib.pyplot as plt
from qiskit import Aer, transpile
from QuantumSchedulers.QAOA.QAOASolver import expected_value
import numpy as np
from JobShopSampler import JobShopSampler, nnull_condition
import os
from Tests.qaoa_basic_benchmarking import test_js_qaoa

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

    # solvers = []
    solvers = [CPLEXSolver(data)]
    # solvers.append(SASolver(data, JobShopHamiltonianConstructor(), 5, variable_pruning=True))
    # solvers.append(QASolver(data, JobShopHamiltonianConstructor(), 8, variable_pruning=True))
    solvers.append(QAOASolver(data, JobShopHamiltonianConstructor(), 5, 1, variable_pruning=True, objective_bias=0,
                              order_bias=1, machine_bias=1, single_run_bias=1, theta=[1, 1]))
    # solvers[0].solve()
    # plottable_solution = solvers[0].get_plottable_solution()
    # solvers[0].plot_solution()
    for solver in solvers[1:]:
        solver.solve(num_reads=1000)
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

def main():

    problem = "Problems/nano_example.txt"
    solution = "micro_example.txt"
    ps = [1, 2]
    solutions = ["Solutions" + os.sep + "res" + str(p) + "_" + solution for p in ps]
    reader = JobShopReader()
    reader.read_problem_data(problem)
    data = reader.get_data()
    test_js_qaoa(problem, solutions, ps)
    plotter = ResultPlotter(solutions[0])
    plotter.plot_solution()
    plt.show()
    return
    qiskit_qaoa = QiskitQAOASolver(data, JobShopHamiltonianConstructor(), 4, 1, variable_pruning=True, objective_bias=0,
                                   theta=[1, 1])
    qaoa = QAOASolver(data, JobShopHamiltonianConstructor(), 4, 1, variable_pruning=True, objective_bias=0,
                      theta=[1, 1])
    qiskit_qaoa.solve()
    #qaoa.solve()
    qiskit_qaoa.plot_solution()
    #qaoa.plot_solution()
    qiskit_qaoa.draw_quantum_circuit()
    qaoa.draw_quantum_circuit()
    plt.show()
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