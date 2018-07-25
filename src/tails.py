"""
Simulate Markov Independent Model
i.e.
Having a Markov source m_iter()

    - Receive integer n
    - Initialize DST T
    - Do n times:  
        - Generate a sequence from m_iter(). Our source therefore needs to be a callable iterator.
        - Stop when this sequence has not been seen previously in the DST
        - Add this new phrase to T


Count tail symbols
This is an addition to the previous algorithm.
When the sequence is to be added to the DST, generate another symbol
and store it into a list of tail symbols : tails[].
"""

from markov import markov_chain
from markov import markov_iter

