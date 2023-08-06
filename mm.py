import numpy as np
from graphviz import Digraph
from typing import List, Optional, Union


class MarkovModel:
    def __init__(self, transition_matrix: Optional[np.ndarray] = None, initial_probabilities: Optional[np.ndarray] = None, states: Optional[List[str]] = None) -> None:
        self.states = states
        self.transition_matrix = transition_matrix
        self.initial_probabilities = initial_probabilities

        
    @property
    def check_transition_matrix(self) -> None:
        assert np.allclose(np.sum(self.transition_matrix, axis=1), np.ones(len(self.transition_matrix))), "Transition matrix rows should sum to 1"

    @property
    def check_states(self) -> None:
        assert len(self.transition_matrix) == len(self.states), "Transition matrix and states names should have the same length"


    def stationary_distribution(self,  n_steps: Optional[int] = None) -> np.ndarray:
        if self.transition_matrix is None:
            raise ValueError("Transition matrix is not provided")

        if not n_steps:   
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
            index = np.argmin(np.abs(eigenvalues - 1))

            stationary_distribution = np.real(eigenvectors[:, index] / np.sum(eigenvectors[:, index]))

            return stationary_distribution
        else:
            return np.linalg.matrix_power(self.transition_matrix, n_steps)

    def train(self, sequence: List[str]) -> None:
        unique_elems = list(set(sequence))

        n_unique_elems = len(unique_elems)

        self.transition_matrix = np.zeros((n_unique_elems, n_unique_elems))

        for i in range(len(sequence) - 1):
            curr_elem = sequence[i]
            next_elem = sequence[i + 1]
            curr_idx = unique_elems.index(curr_elem)
            next_idx = unique_elems.index(next_elem)
            self.transition_matrix[curr_idx, next_idx] += 1

        self.transition_matrix[np.where(~self.transition_matrix.any(axis=1))[0]] = 1
        self.transition_matrix /= self.transition_matrix.sum(axis=1, keepdims=True)

        self.states = unique_elems
        
        
    def sample(self, n_steps: int = 10, current_state: Optional[Union[int, str]] = None, use_names: bool = True) -> List[Union[int, str]]:
        if self.transition_matrix is None:
            raise ValueError("Transition matrix is not provided")
        
        self.check_transition_matrix

        if not self.states:
            self.states = np.arange(len(self.transition_matrix))

        self.check_states

        if not current_state:
            current_state = np.random.choice(np.arange(len(self.transition_matrix)), p = self.initial_probabilities) #list(self.states.values())
        elif type(current_state) == str:
            current_state = self.states.index(current_state)
            
        sequence = [current_state]

        for _ in range(n_steps - 1):
            current_state = np.random.choice(np.arange(len(self.transition_matrix)), p=self.transition_matrix[current_state])
            sequence.append(current_state)

        if use_names is not None:
            return [self.states[i] for i in sequence]
        else:
            return sequence

    
    def __str__(self) -> str:
        return f"MarkovChain(\nstates: \n{self.states} \ntransition_matrix: \n{self.transition_matrix})"
    
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
                    
        dot.render(f'{save_path}', format=file_format, view=view)
        

transition_matrix = np.array([[0.2, 0.6, 0.2],
                             [0.3, 0, 0.7],
                             [0.5, 0, 0.5]])

#Example usage
# mm = MarkovModel(states = ['st 1', 'st 2', 'st 3'], transition_matrix=transition_matrix)
# print(mm)

# print(mm.sample(n_steps = 5))

# print(mm.stationary_distribution())

# print(mm.stationary_distribution(n_steps=1000))

# mm.render()