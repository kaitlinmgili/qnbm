from qiskit import Aer,QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.providers.ibmq.runtime import UserMessenger, ProgramBackend
from qiskit.algorithms.optimizers import NELDER_MEAD, ADAM, GradientDescent
from qiskit.providers.aer import AerSimulator
from qiskit import transpile
import numpy as np
from numpy import pi
import math

class QuantumNeuralNetwork:

    def __init__(self, width_list, k):
        self.structure = width_list
        self.depth = len(width_list)
        self.edge_no = self.EdgeNo() 
        self.neuron_no = sum(width_list) 
        self.ancilla_qubit_no = k 
        self.cbits_per_neuron = 2**k - 1
        self.ancilla_bit_no = (self.neuron_no - self.structure[0]) * (2**k - 1)
        self.answer_bit_no = self.structure[self.depth - 1]
        self.qubit_no = self.neuron_no + self.ancilla_qubit_no 
        self.bit_no = self.ancilla_bit_no + self.answer_bit_no

    def EdgeNo(self): #Defines the number of edges in the model 
        edgeno = 0
        for i in range(len(self.structure)-1):
            edgeno += self.structure[i] * self.structure[i+1]
        return edgeno

    def Normalise(self, weights_unnormalised, biases_unnormalised): #Normalizes the weights and biases of the model 
        weight_counter = 0
        bias_counter = 0
        weights = np.copy(weights_unnormalised)
        biases = np.copy(biases_unnormalised)
        for i in range(self.depth-1):
            current_layer = self.structure[i]
            next_layer = self.structure[i+1]
            normalisation = pi/4 / (current_layer + 1)
            no_of_weights = current_layer * next_layer
            no_of_biases = next_layer
            weights[weight_counter : weight_counter + no_of_weights] = weights[weight_counter : weight_counter + no_of_weights] * normalisation
            biases[bias_counter : bias_counter + no_of_biases] = biases[bias_counter : bias_counter + no_of_biases] * normalisation + pi/4
            weight_counter += no_of_weights
            bias_counter += no_of_biases
        return weights, biases
        
    def CreateCircuit(self, weights_unnorm, biases_unnorm, initialisation = 'h'): #Defines the neruon circuit using the weights and biases 

        circ = QuantumCircuit(self.qubit_no, self.bit_no)
        first_layer_size = self.structure[0]
        weights, biases = self.Normalise(weights_unnorm, biases_unnorm)
        print("Normalized weights and biases: ", weights, biases)
        if initialisation == 'h':
            for i in range(self.ancilla_qubit_no, self.ancilla_qubit_no + first_layer_size):
                circ.h(i)
        else:
            circ = circ.compose(initialisation, range(self.ancilla_qubit_no, self.ancilla_qubit_no + first_layer_size))
        qubit_counter = self.ancilla_qubit_no 
        weight_counter = 0
        bias_counter = 0 
        cbit_counter = 0 
        for i in range(self.depth-1):
            circ, qubit_counter, weight_counter, bias_counter, cbit_counter = self.AddLayer(circ, qubit_counter, weight_counter, bias_counter, cbit_counter, i, weights, biases)
        circ.measure(range(self.qubit_no - self.answer_bit_no, self.qubit_no), range(self.ancilla_bit_no, self.bit_no))
        return circ

    def AddLayer(self, circ, qubit_counter, weight_counter, bias_counter, cbit_counter, i, weights, biases): #Adds a neuron layer to the circuit  
        previous_layer_size = self.structure[i]
        next_layer_size = self.structure[i+1]
        previous_layer = range(qubit_counter, qubit_counter + previous_layer_size)
        next_layer = range(qubit_counter + previous_layer_size, qubit_counter + previous_layer_size + next_layer_size) 

        for output_qubit in next_layer:
            relevant_weights = weights[weight_counter : weight_counter + previous_layer_size]
            relevant_bias = biases[bias_counter]
            circ = self.ConnectLayerToNode(circ, cbit_counter, previous_layer, output_qubit, relevant_weights, relevant_bias, self.ancilla_qubit_no)
            cbit_counter += self.cbits_per_neuron
            weight_counter += previous_layer_size
            bias_counter += 1
        qubit_counter += previous_layer_size
        return circ, qubit_counter, weight_counter, bias_counter, cbit_counter

    def ConnectLayerToNode(self, circ, cbit_counter, previous_layer, output_qubit, weights, bias, k): #Connects the previous layer to a neuron in the next layer  
        if k == 1:
            for i, input_qubit_i in enumerate(previous_layer):
                weight_i = weights[i]
                circ.cry(2*weight_i, input_qubit_i, 0)
            circ.ry(2*bias, 0)
            circ.cy(0, output_qubit)
            circ.rz(-pi/2, 0)
            for i, input_qubit_i in enumerate(previous_layer):
                weight_i = weights[i]
                circ.cry(-2*weight_i, input_qubit_i, 0)
            circ.ry(-2*bias, 0)
            circ.measure(0, cbit_counter)
            circ.reset(0)
        else:
            circ = self.ConnectLayerToNode(circ, cbit_counter, previous_layer, k-1, weights, bias, k-1)
            circ.cy(k-1, output_qubit)
            circ.rz(-pi/2, k-1)
            circ = self.ConnectLayerToNode(circ, cbit_counter + 2**(k-1) - 1, previous_layer, k-1, -1*weights, -1*bias, k-1)
            circ.measure(k-1, cbit_counter + 2**k - 2)
            circ.reset(k-1)
        return circ

    def CreateCircuitList(self, weights_list, biases_list): #Keeps track of the weights and biases in the model 
        list = []
        for i in range(len(weights_list)):
            weights = weights_list[i]
            biases = biases_list[i]
            list.append(self.CreateCircuit(weights,biases))
        return list

    def FindProbs(self, counts, epsilon = 0, datatype = 'dict'): #Converts the circuit counts to an approximate probability distribution 
        new_counts = {}
        probs_dict = {}
        sum = 0
        for count in counts:
            if count[self.bit_no - self.ancilla_bit_no:] == '0'*self.ancilla_bit_no:
                new_counts[count[0:self.answer_bit_no]] = counts[count]
                sum += counts[count]
        for new_count in new_counts:
            probs_dict[new_count] = new_counts[new_count]/sum
        if datatype == 'dict':
            return probs_dict
        else:
            if datatype == 'array':
                return self.ProbsDict_to_ProbsArray(probs_dict, epsilon)
            else:
                print('only dict and array are accepted as types')
                return

    def ProbsDict_to_ProbsArray(self, probs_dict,epsilon): #Converts the distribution dictionary to an array 
        keys = list(probs_dict.keys())
        keyno = len(keys)
        bitno = len(keys[0])
        nocounts = 2**bitno - keyno
        beta = epsilon * nocounts / keyno
        probs_array = np.zeros(2**bitno)
        for i in range(2**bitno):
            i_bin = format(i, '0'+str(bitno)+'b')
            if i_bin in probs_dict:
                probs_array[i] = probs_dict[i_bin] - beta
            else:
                probs_array[i] = epsilon
        return probs_array

    def FindCounts(self, counts): #Gets the correct counts of the output neurons 
        new_counts = {}
        for count in counts:
            if count[self.bit_no - self.ancilla_bit_no:] == '0'*self.ancilla_bit_no:
                new_counts[count[0:self.answer_bit_no]] = counts[count]
        return new_counts
    def KL(self, trained_distribution, target_distribution,epsilon):
        target_distribution = self.ProbsDict_to_ProbsArray(target_distribution, epsilon)
        trained_distribution = self.ProbsDict_to_ProbsArray(trained_distribution, epsilon)
        return np.dot(target_distribution, np.log(target_distribution/trained_distribution))
        
def main(backend: ProgramBackend, user_messenger: UserMessenger, target_distribution, structure, k, initial_weights, initial_biases, maxiter, eps, lr, seed, gradient = "finite"): #Runs the QNBM algorithm 
    epsilon = 1e-16 
    weight_no = len(initial_weights)
    bias_no = len(initial_biases)
    initial_params = np.concatenate([initial_weights, initial_biases])
    qnn = QuantumNeuralNetwork(structure, k)

    iteration_list = []
    params_list = []
    KL_list = []
    
    def runqnn(parameters): 
        shots = 102400
        params_list.append(parameters)
        w = parameters[0:weight_no]
        b = parameters[weight_no:]
        circuit = qnn.CreateCircuit(w,b)
        tcirc = transpile(circuit, backend)

        # Execute noisy simulation and get counts
        result = backend.run(tcirc, shots=shots).result()
        counts = result.get_counts()
        
        probs = qnn.FindProbs(counts, epsilon, 'array')
        KL_list.append(KL(probs))
        iteration_list.append(1)
        if len(iteration_list) % len(parameters) == 0: 
            print(KL(probs))
        return KL(probs)    
       
    def KL(trained_distribution):
        return np.dot(target_distribution, np.log(target_distribution/trained_distribution))
    
    var_bounds = [(-1,1)]*(weight_no+bias_no)
    np.random.seed(seed)
    optimizer = ADAM(maxiter = maxiter, eps=eps, lr = lr) 
    if gradient == "finite":
        result = optimizer.minimize(runqnn,
                       initial_params,
                       bounds = var_bounds)
 
    params_list.append(result.x)
    KL_list.append(result.fun)
    return params_list, KL_list
