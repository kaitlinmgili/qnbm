namespace Testqneuron{
    open Microsoft.Quantum.Intrinsic;     // for H
    open Microsoft.Quantum.Measurement;   // for MResetZ
    open Microsoft.Quantum.Canon;         // for ApplyToEach
    open Microsoft.Quantum.Arrays;        // for ForEach
    open Microsoft.Quantum.Core;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Convert;
   
    @EntryPoint()
    operation createCircuitQ(): (Result[]){
        //1, 0, 2
        let weights = [ 1.34323741, -1.32426818];
        let biases = [0.07816141, 1.41207129];

        let structure = [1, 2];
        let depth = 2;
        let ancilla_qubit_no = 1;
        let neuron_no = 3; //ask about this
        let first_layer_size = structure[0];//1
        let answer_bit_no = structure[depth - 1]; //1
        let qubit_no = neuron_no + ancilla_qubit_no; //3
        
        let num_output = structure[depth - 1]; //1

        use qs = Qubit[qubit_no];
        let end = (ancilla_qubit_no + first_layer_size) - 1; //1
        for qubit in ancilla_qubit_no .. end {
            H(qs[qubit]);
        }
        //All mutables that are needed
        mutable qubit_counter = ancilla_qubit_no; //1
        mutable weight_counter = 0;
        mutable bias_counter = 0;

        for i in 0 .. depth - 2 {
                set (qubit_counter, weight_counter, bias_counter)
                = addLayer(qs, structure, qubit_counter, weight_counter, bias_counter, i, weights, biases, ancilla_qubit_no);
        }
        return MeasureEachZ(qs[qubit_no - answer_bit_no .. (qubit_no-1)]); //2 .. 2
    }

    operation addLayer(qs: Qubit[], structure: Int[], qubit_counter: Int, weight_counter: Int, bias_counter: Int, i: Int, weights: Double[], biases: Double[], ancilla_qubit_no: Int): (Int, Int, Int){
            mutable weight_counter_new = weight_counter; //0
            mutable bias_counter_new = bias_counter; //0
            mutable qubit_counter_new = qubit_counter; //1
            let previous_layer_size = structure[i]; //1
            let next_layer_size = structure[i+1]; //1
            mutable counter = 0;
            let previous_layer = qubit_counter..(qubit_counter+previous_layer_size-1); //1...1
            let next_layer = qubit_counter+previous_layer_size..(qubit_counter + previous_layer_size + next_layer_size - 1); //2...2

            for output_qubit in next_layer{
                 //int that will store the results of the mid-circuit measurements; forcing type to be Int instead of Result -> initialize to 2
                Message($"output_qubit layer: {output_qubit} ");
                mutable succeeded = false;
                for count in 1 .. 7{ //RUS Circuits iterations (for success) are expected to be less than 7
                    if not succeeded{
                        let endPos = weight_counter_new + previous_layer_size - 1; //0
                        let relevant_weights = weights[weight_counter_new .. endPos]; //weights[0]
                        let relevant_bias = biases[bias_counter_new]; //biases[0]
                        let cb = measuringAncilla(qs, previous_layer, output_qubit, relevant_weights, relevant_bias, ancilla_qubit_no);
                        if cb == Zero{
                            set succeeded = true;
                        }
                        else{ //if the mid-measurement for the previous RUS circuit came out to be 1 (RUS did not succeed)... 
                            //Ancilla qubit needs to be reset
                            X(qs[0]);
                            Ry(((PI()*-1.0)/2.0), qs[output_qubit-1]);
                        }
                        //Storing the mid circuit measurement
                        // 2 output neurons
                        // output neuron 1 
                        //1 1 1 1...0 -> 110
                    }
                }
                set weight_counter_new = weight_counter_new + previous_layer_size; //1
                set bias_counter_new = bias_counter_new + 1; //1
                set counter = counter + 1; //1
            }
            set qubit_counter_new = qubit_counter_new + previous_layer_size; //2
            return (qubit_counter_new, weight_counter_new, bias_counter_new);
    }

    operation measuringAncilla(qs: Qubit[], previous_layer: Range, output_qubit: Int, weights: Double[], bias: Double, k: Int): Result{
            connectLayerToNode(qs, previous_layer, output_qubit, weights, bias, k);
            if k == 1{
                let result =  M(qs[0]);
                Reset(qs[0]);
                return result;
            }
            else{
                let result = M(qs[k-1]);
                Reset(qs[k-1]);
                return result;
            }
        
    }

    function Negated(arr : Double[]) : Double[] {   
            mutable negated = [0.0, size=Length(arr)];    
            for idx in IndexRange(arr) {        
                set negated w/= idx <- -arr[idx];    
            }
            return negated;
    }

    operation connectLayerToNode(qs: Qubit[], previous_layer: Range, output_qubit: Int, weights: Double[], bias: Double, k: Int): Unit is Ctl { 
            if k == 1{
                for i in previous_layer{  //1..1
                    let weight_i = weights[i-1]; //weights[0]
                    Controlled Ry([qs[i]], ((weight_i*2.0), qs[0]));
                }
                Ry((bias*2.0), qs[0]);
                Controlled Y([qs[0]], (qs[output_qubit]));
                Rz(((PI()*-1.0)/2.0), qs[0]);
                for i in previous_layer{
                    let weight_i = weights[i-1]; 
                    Controlled Ry([qs[i]], ((weight_i*-2.0), qs[0]));
                }
                Ry((bias*-2.0), qs[0]);
            }
            else{
                connectLayerToNode(qs, previous_layer, k-1, weights, bias, k-1);
                Controlled Y([qs[k-1]], (qs[output_qubit]));
                Rz(((PI()*-1.0)/2.0), qs[k-1]);  
                connectLayerToNode(qs, previous_layer, k-1, Negated(weights), (-1.0*bias), k-1);
            }
    }

    

}
