import numpy as np
from graphviz import Digraph
from typing import List, Optional, Union, Tuple



class HiddenMarkovModel:
    def __init__(self, transition_matrix: np.ndarray, emission_matrix: np.ndarray, initial_probabilities: np.ndarray, states: Optional[List[str]] = None, observations: Optional[List[str]] = None) -> None:
        self.states = states
        self.observations = observations
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.initial_probabilities = initial_probabilities

        self.check_transition_matrix

        if not self.states:
            self.states = np.arange(len(self.transition_matrix))

        if not self.observations:
            self.observations = np.arange(self.emission_matrix.shape[1])

        self.check_states
        self.check_observations

    @property
    def check_transition_matrix(self) -> None:
        assert np.allclose(np.sum(self.transition_matrix, axis=1), np.ones(len(self.transition_matrix))), "Transition matrix rows should sum to 1"

    @property
    def check_states(self) -> None:
        assert len(self.transition_matrix) == len(self.states), "Transition matrix and states names should have the same length"

    @property
    def check_observations(self) -> None:
        assert self.emission_matrix.shape[1] == len(self.observations), "Emission matrix and observations names should have the same length"

    def stationary_distribution(self,  n_steps: Optional[int] = None) -> np.ndarray:
        if not n_steps:   
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
            index = np.argmin(np.abs(eigenvalues - 1))

            stationary_distribution = np.real(eigenvectors[:, index] / np.sum(eigenvectors[:, index]))

            return stationary_distribution
        else:
            return np.linalg.matrix_power(self.transition_matrix, n_steps)

    def sample(self, n_steps: int, current_state: Optional[Union[int, str]] = None) -> Tuple[List[int], List[str]]:
        if not current_state:
            current_state = np.random.choice(len(self.states), p=self.initial_probabilities)
        elif type(current_state) == str:
            current_state = self.states.index(current_state)

        sequence = []

        for _ in range(n_steps):
            sequence.append(np.random.choice(len(self.emission_matrix[current_state]), p=self.emission_matrix[current_state]))
            current_state = np.random.choice(len(self.states), p=self.transition_matrix[current_state])

        sequence_states = [self.observations[i] for i in sequence]
        return sequence, sequence_states
    

    def forward(self, observations: List[int]) -> np.ndarray:
        T = len(observations)
        N = len(self.states)
        alpha = np.zeros((T, N))

        # Initialization
        alpha[0] = self.initial_probabilities * self.emission_matrix[:, observations[0]]

        # Induction
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_matrix[:, j]) * self.emission_matrix[j, observations[t]]

        return alpha

    def backward(self, observations: List[int]) -> np.ndarray:
        T = len(observations)
        N = len(self.states)
        beta = np.zeros((T, N))

        # Initialization
        beta[T-1] = 1.0

        # Induction
        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[t, i] = np.sum(beta[t+1] * self.transition_matrix[i] * self.emission_matrix[:, observations[t+1]])

        return beta

    def forward_backward(self, observations: List[int]) -> Tuple[List[str], np.ndarray]:
        T = len(observations)
        N = len(self.states)
        alpha = self.forward(observations)
        beta = self.backward(observations)
        state_sequence = []

        # for t in range(T):
        #     prob = alpha[t] * beta[t]
        #     prob /= np.sum(prob)
       
        #     state_sequence.append(self.states[np.argmax(prob)])

        posterior = alpha * beta / np.sum(alpha * beta, axis=1, keepdims=True)
     
        for t in range(T):
            state_sequence.append(self.states[np.argmax(posterior[t])])

        return state_sequence, posterior


    def viterbi(self, observations: List[int]) -> List[str]:
        T = len(observations)
        N = len(self.states)
        delta = np.zeros((T, N))
        psi = np.zeros((T, N))

        # Initialization
        delta[0] = self.initial_probabilities * self.emission_matrix[:, observations[0]]

        # Recursion
        for t in range(1, T):
            for j in range(N):
                # delta[t, j] = np.max(delta[t-1] * self.transition_matrix[:, j]) * self.emission_matrix[j, observations[t]]
                # psi[t, j] = np.argmax(delta[t-1] * self.transition_matrix[:, j])

                probs = delta[t-1] * self.transition_matrix[:, j] * self.emission_matrix[:, observations[t]]
                delta[t, j] = np.max(probs)
                psi[t, j] = np.argmax(probs)
                
        # Path backtracking
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(delta[T-1])

        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]

        state_sequence = [self.states[i] for i in path]
        return state_sequence
    

    def baum_welch(self, observations: List[int], n_iterations: int) -> None:
        T = len(observations)
        N = len(self.states)
        M = len(self.observations)
        alpha = self.forward(observations)
        beta = self.backward(observations)
        xi = np.zeros((T - 1, N, N))
        gamma = np.zeros((T, N))

        # for t in range(T):
        #     gamma[t] = alpha[t] * beta[t] / np.sum(alpha[t] * beta[t]) 

        # E-step: compute xi and gamma
        gamma = alpha * beta / np.sum(alpha * beta, axis=1, keepdims=True)

        for _ in range(n_iterations):
            for t in range(T - 1):
                # denominator = np.sum(alpha[t] * beta[t])
                for i in range(N):
                    for j in range(N):
                        xi[t, i, j] = alpha[t, i] * self.transition_matrix[i, j] * self.emission_matrix[j, observations[t + 1]] * beta[t + 1, j] #/ denominator

            xi /= np.sum(xi, axis=(1, 2), keepdims=True)

            # M-step: update transition matrix, emission matrix, and initial probabilities
            self.transition_matrix = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0, keepdims=True)

            self.emission_matrix = np.zeros((N, M))
            for j in range(N):
                for k in range(M):
                    mask = np.array(observations) == k
                    self.emission_matrix[j, k] = np.sum(gamma[:, j] * mask) / np.sum(gamma[:, j])
            # self.emission_matrix = np.sum(posterior[:, :, np.newaxis] * (observations == np.arange(len(self.observations))[:, np.newaxis]), axis=0) / np.sum(gamma, axis=0)

            self.initial_probabilities = gamma[0]

            self.transition_matrix /= self.transition_matrix.sum(axis=1, keepdims=True)
            self.emission_matrix /= self.emission_matrix.sum(axis=1, keepdims=True)
            # self.initial_probabilities /= self.initial_probabilities.sum()

    def render(self, view: bool = True, save_path: str = "markov-chain-viz", file_format: str = "png") -> None:
        dot = Digraph()

        n_states = len(self.states)
        for i in range(n_states):
            dot.node(str(i), label=str(self.states[i]))

        for i in range(n_states):
            for j in range(n_states):
                weight = self.transition_matrix[i, j]
                if weight > 0:
                    dot.edge(str(i), str(j), label=str(weight))

        n_observes = self.emission_matrix.shape[1]
        for i in range(n_observes):
            dot.node(str(i+n_states), label=str(self.observations[i]), color='red')

        for i in range(n_states):
            for j in range(n_observes):
                emission = self.emission_matrix[i, j]
                if emission > 0:
                    # dot.node_attr.update({'label': f'{self.states[i]}\nEmission: {emission}'})
                    # dot.edge_attr.update({'label': f'{self.states[j]}'})
                    dot.edge(str(i), str(j+n_states), label=str(emission), color='red')

                    
        dot.render(f'{save_path}', format=file_format, view=view)


# # Example usage
# states = ['Sunny', 'Rainy']
# transition_matrix = np.array([[0.8, 0.2], [0.4, 0.6]])
# emission_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
# initial_probabilities = np.array([0.6, 0.4])

# hmm = HiddenMarkovModel(transition_matrix, emission_matrix, initial_probabilities, states)
# sequence = hmm.sample(10)
# print("Generated Sequence:", sequence)
# predicted_states = hmm.forward_backward(sequence)
# print("Predicted States:", predicted_states)


# transition_matrix = np.array([[0.7, 0.3], [0.3, 0.7]])
# emission_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])

# states = ['Sunny', 'Rainy']
# initial_probabilities = np.array([0.5, 0.5])

# mm = HiddenMarkovModel(transition_matrix, emission_matrix, initial_probabilities, states)
# sequence = mm.sample(2)

# print(sequence)

# predicted_states, posterior = mm.forward_backward(sequence)
# print(posterior)
# print(predicted_states)


transition_matrix = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6]])
emission_matrix = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])
initial_probabilities = np.array([0.4, 0.3, 0.3])

weather_model = HiddenMarkovModel(transition_matrix, emission_matrix, initial_probabilities, states=["Солнечная погода", "Облачная погода", "Дождливая погода"], observations=["Плохое настроение", "Хорошее настроение"])
# weather_model.render()

# Генерация последовательности состояний
sequence, observations = weather_model.sample(n_steps=5)
print("Сгенерированная последовательность состояний:", observations)

# Восстановление наиболее вероятной последовательности состояний
state_sequence, posterior = weather_model.forward_backward(sequence)
print("Наиболее вероятная последовательность состояний:", state_sequence)
print(posterior)



import numpy as np
"""https://neerc.ifmo.ru/wiki/index.php?title=%D0%90%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_%22%D0%92%D0%BF%D0%B5%D1%80%D0%B5%D0%B4-%D0%9D%D0%B0%D0%B7%D0%B0%D0%B4%22"""

def forward_backward(transition_probability, emit_probability, initial_probabilities, observations):
    num_states = transition_probability.shape[0]
    num_observations = len(observations)

    fwd = np.zeros((num_states, num_observations))
    bkw = np.zeros((num_states, num_observations))
    probabilities = np.zeros((num_states, num_observations))

    # Forward algorithm
    def alpha(s, t):
        if t == 0:
            return emit_probability[s, observations[t]] * initial_probabilities[s]
        if fwd[s, t] != 0:
            return fwd[s, t]
        f = np.sum(alpha(j, t - 1) * transition_probability[j, s] for j in range(num_states))
        f *= emit_probability[s, observations[t]]
        fwd[s, t] = f
        return fwd[s, t]

    # Backward algorithm
    def beta(s, t):
        if t == num_observations - 1:
            return 1
        if bkw[s, t] != 0:
            return bkw[s, t]
        b = np.sum(beta(j, t + 1) * transition_probability[s, j] * emit_probability[j, observations[t + 1]] for j in range(num_states))
        bkw[s, t] = b
        return bkw[s, t]

    # Compute forward and backward probabilities
    for s in range(num_states):
        fwd[s, 0] = emit_probability[s, observations[0]] * initial_probabilities[s]
        bkw[s, num_observations - 1] = 1

    chain_probability = np.sum(alpha(j, 0) * beta(j, 0) for j in range(num_states))

    # Compute probabilities
    for s in range(num_states):
        for t in range(num_observations):
            probabilities[s, t] = (alpha(s, t) * beta(s, t)) / chain_probability

    return probabilities



# predicted_states = forward_backward(transition_matrix, emission_matrix, initial_probabilities, sequence)
# # print(posterior)
# print(predicted_states)

predicted_states = weather_model.viterbi(sequence)
print(predicted_states)