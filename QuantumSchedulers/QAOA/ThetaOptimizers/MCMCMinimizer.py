from QuantumSchedulers.QAOA.QAOA import ThetaOptimizer, QCSampler, CircuitBuilder
from QuantumSchedulers.QAOA.ThetaOptimizers.QiskitMinimizer import get_expectation, key_to_vector
from scipy.optimize import minimize
import numpy as np
from typing import Tuple, List
from scipy.stats import norm, uniform
from scipy.stats._continuous_distns import norm_gen
import math
from random import random
import matplotlib.pyplot as plt
import copy
import matplotlib.pyplot as plt
import seaborn as sns


class MCMCMinimizer(ThetaOptimizer):
    def __init__(self, method: str="COBYLA", beta=5, xi=0.5, epsilon=0.05, eta=0.1, mcmc_steps=1000):
        super().__init__()
        self._method = method
        self._beta = beta
        self._xi = xi
        self._epsilon = epsilon
        self._eta = eta
        self._mcmc_steps = mcmc_steps
        self._bounds = None

        self._lambda_t = None
        self._sigma2_t = None
        self._grad_lambda_t = None
        self._proposal_distributions_t = None
        self._lambdas = list()
        self._steps = 0
        self._theta_t = None

    def optimize_theta(self, circuit_builder: CircuitBuilder, qc_sampler: QCSampler, num_reads: int, hamiltonian,
                       theta, verbose=False):
        self._circuit_builder = circuit_builder
        self._qc_sampler = qc_sampler
        self._num_reads = num_reads
        self._hamiltonian = hamiltonian
        theta = np.array(theta)
        self._C = 0.5 * np.sum(np.diag(self._hamiltonian)) \
                  + np.sum(self._hamiltonian - np.diag(np.diag(self._hamiltonian))) / 4
        for i in range(3):
            print([(self._hamiltonian[i, j] + self._hamiltonian[j, i])
                                                for j in range(3)])
        """
        Test case for lambda_t
        shape = (32, 32)
        beta_stepsize = math.pi / shape[0]
        gamma_stepsize = 2 * math.pi / shape[1]
        result = np.zeros(shape)
        for i, j in np.ndindex(shape):
            result[i, j] = self.compute_lambda([(shape[0] - i) * beta_stepsize, j * gamma_stepsize]) \
                           + 0.5*np.sum(np.diag(self._hamiltonian)) + np.sum(self._hamiltonian - np.diag(np.diag(self._hamiltonian)))/4
            if j == 0:
                print(i)
            #print((shape[0] - i) * beta_stepsize, j * gamma_stepsize, result[i, j])
        fig, ax = plt.subplots()
        ax = sns.heatmap(result, center=0, xticklabels=['0', r'$\pi$', r'$2\pi$'], yticklabels=[r'$\pi$', '0'])
        ax.set_title(r'$F_1(\mathbf{\gamma},\mathbf{\beta})$')
        ax.set_xlabel(r'$\mathbf{\gamma}$')
        ax.set_ylabel(r'$\mathbf{\beta}$')
        ax.set_xticks([0, shape[1] / 2, shape[1]])
        ax.set_yticks([0, shape[0]])
        plt.show()
        """

        self._bounds = [(0, math.pi) if k < theta.size / 2 else (0, 2 * math.pi) for k in range(theta.size)]
        self.mcmc_init(theta)
        for i in range(self._mcmc_steps):
            print("Theta ", self._steps, self._theta_t, "Lambda_t", self._lambda_t, "Grad_lambda_t", self._grad_lambda_t,
                  "Sigma2_t", self._sigma2_t)
            self.mcmc_step()
            if len(self._lambdas) > 0 and self._lambda_t == self._lambdas[len(self._lambdas)-1]:
                continue
            self._lambdas.append(self._lambda_t)
            self._steps += 1
            if self._steps > 1000:
                self._lambdas.pop(0)
            if self._steps % 1000 == 0 and self._steps >= 1000:
                sns.histplot(self._lambdas)
                plt.show()

        expectation = get_expectation(self._hamiltonian, self._circuit_builder, self._qc_sampler, self._num_reads)

        res = minimize(expectation, self._theta_t, method=self._method)
        self._theta = res.x
        self._expected_energy = res.fun
        if verbose:
            print(res)

    def get_name(self):
        return "MCMC_QAOA"

    def mcmc_init(self, theta_init):
        self._lambda_t, self._sigma2_t = self.compute_total_lambda_and_sigma2(theta_init)
        self._grad_lambda_t = self.compute_grad_lambda(theta_init)
        self._proposal_distributions_t = self.get_proposal_distributions(self._grad_lambda_t, self._sigma2_t)
        self._theta_t = theta_init

    def mcmc_step(self):

        theta_d = self._theta_t - np.array([dist.rvs() for dist in self._proposal_distributions_t])
        """
        #non periodic boundary
        for k in range(theta_d.size):
            if theta_d[k] < self._bounds[k][0]:
                theta_d[k] = self._bounds[k][0]
            elif theta_d[k] > self._bounds[k][1]:
                theta_d[k] = self._bounds[k][1]
        """
        lambda_d, sigma2_d = self.compute_total_lambda_and_sigma2(theta_d)
        grad_lambda_d = self.compute_grad_lambda(theta_d)
        proposal_distributions_d = self.get_proposal_distributions(grad_lambda_d, sigma2_d)
        a = self.acceptance_distribution(theta_d, lambda_d, proposal_distributions_d)
        u = uniform.rvs()
        print("Acceptance probability: ", a, " Lambda_d: ", lambda_d)
        if a > u:
            self._theta_t = theta_d
            self._lambda_t = lambda_d
            self._sigma2_t = sigma2_d
            self._grad_lambda_t = grad_lambda_d
            self._proposal_distributions_t = proposal_distributions_d

    def acceptance_distribution(self, theta_d: np.ndarray, lambda_d: float,
                                proposal_distributions_d: List[norm_gen]):

        g_d_given_t = np.prod([self._proposal_distributions_t[k].pdf(self._theta_t[k] - theta_d[k]) for k in range(theta_d.size)])
        g_t_given_d = np.prod([proposal_distributions_d[k].pdf(theta_d[k] - self._theta_t[k]) for k in range(theta_d.size)])
        print("Ratio", g_t_given_d/g_d_given_t)
        #print([proposal_distributions_t[k].pdf(theta_t[k] - theta_d[k]) for k in range(theta_d.size)])
        #print([proposal_distributions_d[k].pdf(theta_d[k] - theta_t[k]) for k in range(theta_t.size)])
        return np.minimum(1, self.p_boltzmann(lambda_d) * g_t_given_d / (self.p_boltzmann(self._lambda_t) * g_d_given_t))

    def get_proposal_distributions(self, grad_lambda: np.ndarray, sigma2: float) -> List[norm_gen]:
        return [norm(self._eta * grad_lambda[k], self._xi**2 + self._eta**2 * sigma2 / (2 * self._epsilon**2))
                for k in range(grad_lambda.size)]

    def p_boltzmann(self, x: float) -> float:
        return math.exp(-self._beta * x)

    def compute_total_lambda_and_sigma2(self, theta: np.ndarray) -> Tuple[float, float]:
        counts = self.execute_circ(theta)
        mu_raw = self.compute_mu_raw(counts)
        mu = self.compute_mu_weighted(mu_raw)
        sigma2 = self.compute_sigma2(mu_raw)
        #print(sigma2)
        return np.sum(mu) + self._C, np.sum(sigma2)/self._num_reads

    def compute_grad_lambda(self, theta: np.ndarray) -> np.ndarray:
        return np.array([self.compute_grad_lambda_k(theta, k) for k in range(theta.size)])

    def compute_grad_lambda_k(self, theta: np.ndarray, k: int):
        epsilon_k_hat = self._epsilon * np.eye(1, theta.size, k)[0]
        return (self.compute_lambda(theta + epsilon_k_hat) - self.compute_lambda(theta - epsilon_k_hat)) \
               / (2 * self._epsilon)

    def compute_lambda(self, theta):
        counts = self.execute_circ(theta)
        mu_raw = self.compute_mu_raw(counts)
        mu = self.compute_mu_weighted(mu_raw)
        return np.sum(mu) + 0.5*np.sum(np.diag(self._hamiltonian)) \
               + np.sum(self._hamiltonian - np.diag(np.diag(self._hamiltonian)))/4

    def compute_mu_weighted(self, mu_raw) -> np.ndarray:
        n = mu_raw.shape[0]
        ising_hamiltonian: np.ndarray = copy.deepcopy(self._hamiltonian) / 4
        for i in range(n):
            ising_hamiltonian[i, i] = -np.sum([(self._hamiltonian[i, j] + self._hamiltonian[j, i])
                                              for j in range(n)]) / 4
        return np.multiply(ising_hamiltonian, mu_raw)

    #diagonal are h, upper triangular are Jij
    def compute_mu_raw(self, counts: dict) -> np.ndarray:
        mu_raw = np.zeros(self._hamiltonian.shape)
        for bitstring, count in counts.items():
            x = key_to_vector(bitstring)
            s = np.ones(len(x)) - 2 * x
            mu_raw_cnt = np.outer(s, s) + np.diag(s) - np.diag(np.ones(len(x))) #remove ones from diagonal as si^2=1
            mu_raw += count * mu_raw_cnt

        return mu_raw / self._num_reads

    def compute_sigma2(self, mu_raw: np.ndarray) -> np.ndarray:
        n = mu_raw.shape[0]
        ising_hamiltonian: np.ndarray = self._hamiltonian / 4
        for i in range(n):
            ising_hamiltonian[i, i] = -np.sum([(self._hamiltonian[i, j] + self._hamiltonian[j, i])
                                              for j in range(n)]) / 4
        return np.multiply(np.multiply(ising_hamiltonian, ising_hamiltonian),
                           np.ones(mu_raw.shape) - np.multiply(mu_raw, mu_raw))

    def execute_circ(self, theta: np.ndarray) -> dict:
        qc = self._circuit_builder.get_quantum_circuit(theta)
        return self._qc_sampler.get_counts(qc, self._num_reads)


def default_init_theta(p):
    return [math.pi * (1 + int(i/p)) * random() for i in range(2*p)]