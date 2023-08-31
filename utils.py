import numpy as np
import pandas as pd 
import typing
import math


def counts_to_prob(self, samples: typing.Dict): #Turns a dictionary of sample counts into probabilities.
    total_shots = np.sum(list(samples.values()))
    for bitstring, count in samples.items(): 
        samples[bitstring] = count / total_shots 
    return samples 

def dec2bin(number: int, length: int) -> typing.List[int]: #Turns a decimal number into a binary bitstring.
    bit_str = bin(number)
    bit_str = bit_str[2 : len(bit_str)]  
    bit_string = [int(x) for x in list(bit_str)]
    if len(bit_string) < length:
        len_zeros = length - len(bit_string)
        bit_string = [int(x) for x in list(np.zeros(len_zeros))] + bit_string
    return bit_string

def comb(n: int, k: int)-> int: #Combinatorial method for n choose k.
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)


