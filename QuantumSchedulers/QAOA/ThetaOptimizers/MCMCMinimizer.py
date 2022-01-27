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


class MCMCMinimizer(ThetaOptimizer):
    def __init__(self, method: str="COBYLA", beta=0.2, xi=0.7, epsilon=0.1, eta=0.1, mcmc_steps=1000):
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

    def optimize_theta(self, circuit_builder: CircuitBuilder, qc_sampler: QCSampler, num_reads: int, hamiltonian,
                       theta, verbose=False):
        self._circuit_builder = circuit_builder
        self._qc_sampler = qc_sampler
        self._num_reads = num_reads
        self._hamiltonian = hamiltonian
        theta = np.array(theta)

        self._bounds = [(0, math.pi) if k < theta.size / 2 else (0, 2 * math.pi) for k in range(theta.size)]
        for i in range(self._mcmc_steps):
            print("Theta ", i, theta)
            theta = self.mcmc_step(theta)

        expectation = get_expectation(self._hamiltonian, self._circuit_builder, self._qc_sampler, self._num_reads)

        res = minimize(expectation, theta, method=self._method)
        self._theta = res.x
        self._expected_energy = res.fun
        if verbose:
            print(res)

    def get_name(self):
        return "MCMC_QAOA"

    def mcmc_step(self, theta_t):
        if self._lambda_t is None:
            self._lambda_t, self._sigma2_t = self.compute_total_lambda_and_sigma2(theta_t)
            self._grad_lambda_t = self.compute_grad_lambda(theta_t)
            self._proposal_distributions_t = self.get_proposal_distributions(self._grad_lambda_t, self._sigma2_t)
        print("Lambda_t", self._lambda_t)
        print("Grad_lambda_t", self._grad_lambda_t)
        #proposed new theta


        theta_d = theta_t + np.array([dist.rvs() for dist in self._proposal_distributions_t])
        for k in range(theta_d.size):
            if theta_d[k] < self._bounds[k][0]:
                theta_d[k] += self._bounds[k][1]
            elif theta_d[k] > self._bounds[k][1]:
                theta_d[k] -= self._bounds[k][1]
        lambda_d, sigma2_d = self.compute_total_lambda_and_sigma2(theta_d)
        grad_lambda_d = self.compute_grad_lambda(theta_d)
        proposal_distributions_d = self.get_proposal_distributions(grad_lambda_d, sigma2_d)
        a = self.acceptance_distribution(theta_d, theta_t, lambda_d, self._lambda_t, proposal_distributions_d,
                                         self._proposal_distributions_t)
        u = uniform.rvs()
        theta_t_next = theta_t
        print("Acceptance probability: ", a, " Lambda_d: ", lambda_d)
        if a > u:
            theta_t_next = theta_d
            self._lambda_t = lambda_d
            self._sigma2_t = sigma2_d
            self._grad_lambda_t = grad_lambda_d
            self._proposal_distributions_t = proposal_distributions_d

        return theta_t_next

    def acceptance_distribution(self, theta_d: np.ndarray, theta_t: np.ndarray, lambda_d: float, lambda_t: float,
                                proposal_distributions_d: List[norm_gen], proposal_distributions_t: List[norm_gen]):

        g_d_given_t = np.prod([proposal_distributions_t[k].pdf(theta_t[k] - theta_d[k]) for k in range(theta_d.size)])
        g_t_given_d = np.prod([proposal_distributions_d[k].pdf(theta_d[k] - theta_t[k]) for k in range(theta_t.size)])

        return np.minimum(1, self.p_boltzmann(lambda_d) * g_t_given_d / (self.p_boltzmann(lambda_t) * g_d_given_t))

    def get_proposal_distributions(self, grad_lambda: np.ndarray, sigma2: float) -> List[norm_gen]:
        return [norm(self._eta * grad_lambda[k], self._xi**2 + self._eta**2 * sigma2 / (2 * self._epsilon**2))
                for k in range(grad_lambda.size)]

    def p_boltzmann(self, x: float) -> float:
        return np.exp(-self._beta * x)

    def compute_total_lambda_and_sigma2(self, theta: np.ndarray) -> Tuple[float, float]:
        counts = self.execute_circ(theta)
        mu_raw = self.compute_mu_raw(counts)
        mu = self.compute_mu_weighted(mu_raw)
        sigma2 = self.compute_sigma2(mu_raw)
        return np.sum(mu), np.sum(sigma2)/self._num_reads

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
        return np.sum(mu)

    def compute_mu_weighted(self, mu_raw) -> np.ndarray:
        n = mu_raw.shape[0]
        ising_hamiltonian: np.ndarray = copy.deepcopy(self._hamiltonian) / 4
        for i in range(n):
            ising_hamiltonian[i, i] = np.sum([0.5 * (self._hamiltonian[i, j] + self._hamiltonian[j, i])
                                              for j in range(n)])
        return np.multiply(ising_hamiltonian, mu_raw)

    #diagonal are h, upper triangular are Jij
    def compute_mu_raw(self, counts: dict) -> np.ndarray:
        mu_raw = np.zeros(self._hamiltonian.shape)
        lbda = 0
        for bitstring, count in counts.items():
            x = key_to_vector(bitstring)
            s = np.ones(len(x)) - 2 * x
            mu_raw_cnt = np.outer(s, s) \
                     + np.diag(s) - np.diag(np.ones(len(x))) #remove ones from diagonal as si^2=1
            mu_raw += count * mu_raw_cnt
            lbda += np.dot(np.dot(x, self._hamiltonian), x)*count

        return mu_raw / self._num_reads

    def compute_sigma2(self, mu_raw: np.ndarray) -> np.ndarray:
        n = mu_raw.shape[0]
        ising_hamiltonian: np.ndarray = self._hamiltonian / 4
        for i in range(n):
            ising_hamiltonian[i, i] = np.sum([0.5 * (self._hamiltonian[i, j] + self._hamiltonian[j, i])
                                              for j in range(n)])
        return np.multiply(np.multiply(ising_hamiltonian, ising_hamiltonian),
                           np.ones(mu_raw.shape) - np.multiply(mu_raw, mu_raw))

    def execute_circ(self, theta: np.ndarray) -> dict:
        qc = self._circuit_builder.get_quantum_circuit(theta)
        return self._qc_sampler.get_counts(qc, self._num_reads)


def default_init_theta(p):
    return [math.pi * (1 + int(i/p)) * random() for i in range(2*p)]