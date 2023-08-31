import numpy as np
import pandas as pd 
import typing
from utils import dec2bin, comb

def subset_distribution( #Generates a cardinality/hamming training probability distribution in the form of a dictionary
    n_qubits: int, 
    numerality: int
    ) -> typing.Dict: 
    dict = {}
    prob = 1 / comb(n_qubits, numerality)
    for i in range(2**n_qubits):
        i_bin = format(i, '0'+str(n_qubits)+'b')
        counter = 0
        for bit in i_bin:
            if bit == '1':
                counter += 1
        if counter == numerality:
            dict[i_bin] = prob
    return dict