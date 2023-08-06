import numpy as np
import sys
sys.path.append(".")

from src.hmm import HiddenMarkovModel

# Example usage
# Define your transition matrix, emission matrix, initial probabilities, states, and observations

transition_matrix = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6]])
emission_matrix = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])
initial_probabilities = np.array([0.4, 0.3, 0.3])

hmm = HiddenMarkovModel(transition_matrix, emission_matrix, initial_probabilities, states=["Sunny weather", "Cloudy weather", "Rainy weather"], observations=["Bad mood", "Good mood"])
hmm.render()

sequence, observations = hmm.sample(n_steps=5)
print("Generated sequence of states:\n", observations)

state_sequence, posterior = hmm.forward_backward(sequence)
print("Restoring the most probable sequence of states via 'forward-backward':\n", state_sequence)
print(posterior)

state_sequence = hmm.viterbi(sequence)
print("Restoring the most probable sequence of states via 'viterbi':\n", state_sequence)

# Perform the Baum-Welch algorithm on the given observation sequence
hmm.baum_welch(sequence, n_iterations=10)

# Updated parameters after running the Baum-Welch algorithm
print("Updated Transition Matrix:")
print(hmm.transition_matrix)
print("Updated Emission Matrix:")
print(hmm.emission_matrix)
print("Updated Initial Probabilities:")
print(hmm.initial_probabilities)