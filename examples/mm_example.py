import numpy as np
import re
import sys
sys.path.append(".")

from mm import MarkovModel



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


