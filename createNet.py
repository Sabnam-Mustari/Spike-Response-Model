import numpy as np
import functools
from neurons import spike
#from WBC_dataset import  IH_weight, HO_weight
from  gussian_new import intersection, trainingSet, IH_weight, HO_weight

class FeedForwardNetwork:
    """
    Fully connected FeedForward Spiking Neural network with 9 neurons in input layer and 8 neurons in hiddens layer.
    SRM model has been developed as model of Spiking neuron  adapted from (http://neurons.readthedocs.io/en/latest/topics/spiking.html).
    This example is provided below with Wisconsin Breast cancer(WBC) dataset provided by UCI.

    """

    def __init__(self, w_12, w_23):
        """
        :param w_12: weight between input to hidden layer
        :paramw_23: weight between hidden to output layer
        :param l3_spike: spike of hidden layer must be all 0 for first time
        """
        # initialize parameters
        self.w_12 = w_12
        self.w_23 = w_23
        #self.l3_spike = l3_spike


    def testing(self, input):

        error = 0.0
        spiketrain_1_2 = trainingSet(i)  # receive spike (training dataset for input neuron(9) and target dataset fot output neuron(1))
        l1_spike = spiketrain_1_2[:9, :]  # Training spike time for input Layer
        targets = spiketrain_1_2[9:, :]  # Target spike time for input Layer
        l2_spike = np.zeros((8, 14))  # No spike for hidden Layer neurons(8) at first
        spiketrain_1_2 = np.concatenate((l1_spike, l2_spike))  # Tatal spike train for input and hidden layer
        #print("Input Spiketrain :\n", spikeTrain, sep='\n')
        print("Input Spiketrain from input to hidden layer:\n", spiketrain_1_2, sep='\n')

        potential = self.feedForward(spiketrain_1_2)


    def training(self, input):

        error = 0.0
        spiketrain_1_2 = trainingSet(i) # receive spike (training dataset for input neuron(9) and target dataset fot output neuron(1))
        l1_spike = spiketrain_1_2[:int(input_Neuron), :]  # Training spike time for input Layer, int(input_Neuron)=9
        targets = spiketrain_1_2[int(input_Neuron):, :]   # Target spike time for input Layer
        l2_spike = np.zeros((int(hidden_Neuron), 14))  # No spike for hidden Layer neurons(8) at first, int(hidden_Neuron)=8
        spiketrain_1_2 = np.concatenate((l1_spike, l2_spike)) # Tatal spike train for input and hidden layer
        print("Targets Pattern:\n", targets, sep='\n')
        print("Input Spiketrain Pattern from input to hidden layer:\n", spiketrain_1_2, sep='\n')

        potential = self.feedForward(spiketrain_1_2)
        error += self.backPropagation(targets, potential)


    def feedForward(self, spiketrain_1_2):

        ######----------- Start Input to Hidden Layer----------######

        srm_model = spike.SRM(neurons=int(hidden_Neuron) + int(input_Neuron)) # (9 + 8 = 17)
        models = [srm_model]

        for model in models:

            neurons, timesteps = spiketrain_1_2.shape
            for t in range(timesteps):
                total_current_1 = model.check_spikes(spiketrain_1_2, self.w_12, t)
        print("Output Spiketrain from input to hidden layer:\n", spiketrain_1_2, sep='\n')

        ######----------- End Input to Hidden Layer----------######


        ######----------- Strat Hidden to Output Layer----------######
        s = spiketrain_1_2[int(input_Neuron):, :].copy()  # Output spike of L_2 Neurons, int(input_Neuron)=9
        l3_spike = np.zeros((int(output_Neuron), 14))  # 2 neurons in Output Layer Neurons
        spiketrain_2_3 = np.concatenate((s, l3_spike))  #
        print("Input Spiketrain from Hidden to Output layer:\n", spiketrain_2_3, sep='\n')

        srm_model = spike.SRM(neurons=int(hidden_Neuron)+int(output_Neuron))
        models = [srm_model]

        for model in models:

            neurons, timesteps = spiketrain_2_3.shape
            for t in range(timesteps):
                total_current_2 = model.check_spikes(spiketrain_2_3, self.w_23, t)
        print("Output Spiketrain from Hidden to Output layer:\n", spiketrain_2_3, sep='\n')

        return total_current_2

    ######----------- End Hidden to Output Layer Processing----------######
    def backPropagation(self, targets, potential):
        MSE = 0
        weight = MSE
        return weight



if __name__ == '__main__':

    # create network
    input_Neuron = input("Enter the number of neuron in Input layer 9:\n ")
    hidden_Neuron = input("Enter the number of neuron in Hidden layer 8:\n ")
    output_Neuron = input("Enter the number of neuron in output layer 2:\n ")
    print('Neuron in input layer, hidden layer and output layer:',input_Neuron, hidden_Neuron, output_Neuron)
    w_12 = IH_weight(int(input_Neuron), int(hidden_Neuron)) # (9x8) Matrix
    w_23 = HO_weight(int(hidden_Neuron),int(output_Neuron)) # (8x2) Matrix
    #w_23 = np.random.random_integers(10, 100, size=(10, 10))  # 8 + 2 = 10 neurons
    #w_23 = (w_23 + w_23.T) / 2  # symmetric matrix
    # print("Weight between Hidden to Output layer:", w_23, sep='\n')
    #l3_spike = np.zeros((int(output_Neuron), 10))  # 1 neurons in Output Layer

    ffn = FeedForwardNetwork(w_12, w_23)

    for i in range(1):
        if i < 1:
            ffn.training(i)
        else:
            ffn.testing(i)
        i += 1
