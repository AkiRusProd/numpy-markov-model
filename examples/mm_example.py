import numpy as np
import re
import sys
sys.path.append(".")

from mm import MarkovModel


# Example usage 1

#Prepare data
with open("data/model-speech.txt", 'r') as f:
    corpus = f.read()
    
corpus = corpus.replace('\n', ' ').lower()
corpus = re.sub(r'[,.\"\'!@#$%^&*(){}?/;`~:<>+=-\\]', '', corpus)

corpus_elems = corpus.split(" ")
unique_elems = list(set(corpus_elems))

n_unique_elems = len(unique_elems)


#Initialize Markov model
nlp_mm = MarkovModel()

#Train Markov model
nlp_mm.train(corpus_elems)

#Sample
sequence = ' '.join(nlp_mm.sample(250, current_state='users'))

print(sequence)



# Example usage 2
# transition_matrix = np.array([[0.2, 0.6, 0.2], [0.3, 0, 0.7], [0.5, 0, 0.5]])
# mm = MarkovModel(states = ['State 1', 'State 2', 'State 3'], transition_matrix=transition_matrix)

# print(mm.sample(n_steps = 5))
# print(mm.stationary_distribution())
# print(mm.stationary_distribution(n_steps=1000))

# mm.render()


