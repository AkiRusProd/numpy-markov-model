import numpy as np
from graphviz import Digraph


class MarkovModel:
    def __init__(self, transition_matrix, initial_probabilities=None, states=None):
        self.states = states
        self.transition_matrix = transition_matrix
        self.initial_probabilities = initial_probabilities

        self.check_transition_matrix

        if not self.states:
            self.states = np.arange(len(self.transition_matrix))

        self.check_states

        
    @property
    def check_transition_matrix(self):
        assert np.allclose(np.sum(self.transition_matrix, axis=1), 1.0), "Transition matrix rows should sum to 1"

    @property
    def check_states(self):
        assert len(self.transition_matrix) == len(self.states), "Transition matrix and states names should have the same length"


    def stationary_distribution(self, n_steps=None):
        if not n_steps:   
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
            index = np.argmin(np.abs(eigenvalues - 1))

            stationary_distribution = np.real(eigenvectors[:, index] / np.sum(eigenvectors[:, index]))

            return stationary_distribution
        else:
            return np.linalg.matrix_power(self.transition_matrix, n_steps)
        
    def sample(self, n_steps=10, current_state=None, use_names=True):
        if not current_state:
            current_state = np.random.choice(np.arange(len(self.transition_matrix)), p = self.initial_probabilities) #list(self.states.values())
            
        sequence = [current_state]

        for _ in range(n_steps - 1):
            current_state = np.random.choice(np.arange(len(self.transition_matrix)), p=self.transition_matrix[current_state])
            sequence.append(current_state)

        if use_names is not None:
            return [self.states[i] for i in sequence]
        else:
            return sequence

    
    def __str__(self):
        return f"MarkovChain(\nstates: \n{self.states} \ntransition_matrix: \n{self.transition_matrix})"
    
    def render(self, view = True, save_path = "markov-chain-viz", file_format = "png"):
        dot = Digraph()

        n_states = len(self.states)
        for i in range(n_states):
            dot.node(str(i), label=str(self.states[i]))

        for i in range(n_states):
            for j in range(n_states):
                weight = self.transition_matrix[i, j]
                if weight > 0:
                    dot.edge(str(i), str(j), label=str(weight))
                    
        dot.render(f'{save_path}', format=file_format, view=view)
        


# transition_matrix = np.array([[0.7, 0.2, 0.1],
#                              [0.3, 0.4, 0.3],
#                              [0.5, 0.1, 0.4]])

transition_matrix = np.array([[0.2, 0.6, 0.2],
                             [0.3, 0, 0.7],
                             [0.5, 0, 0.5]])

mm = MarkovModel(states = ['st 1', 'st 2', 'st 3'], transition_matrix=transition_matrix)
print(mm)

print(mm.sample(n_steps = 5))

print(mm.stationary_distribution())

# print(mc.stationary_distribution(n_steps=1000))

# mm.render()