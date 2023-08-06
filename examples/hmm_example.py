import numpy as np
import sys
sys.path.append(".")

from hmm import HiddenMarkovModel





# Usage example
# Define your transition matrix, emission matrix, initial probabilities, states, and observations
transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
emission_matrix = np.array([[0.2, 0.4, 0.4], [0.5, 0.4, 0.1]])
initial_probabilities = np.array([0.6, 0.4])
states = ['Sunny', 'Rainy']
observations = ['Dry', 'Dryish', 'Damp']

hmm = HiddenMarkovModel(transition_matrix, emission_matrix, initial_probabilities, states, observations)
observations_sequence = [0, 1, 2, 2, 0]  # Index values corresponding to 'Dry', 'Dryish', 'Damp', etc.

# Perform the Baum-Welch algorithm on the given observation sequence
hmm.baum_welch(observations_sequence, n_iterations=10)

# Updated parameters after running the Baum-Welch algorithm
print("Updated Transition Matrix:")
print(hmm.transition_matrix)
print("Updated Emission Matrix:")
print(hmm.emission_matrix)
print("Updated Initial Probabilities:")
print(hmm.initial_probabilities)