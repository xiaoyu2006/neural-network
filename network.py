import numpy as np

def sigmoid(input):
    tmp=1+np.exp(input)
    return 1/tmp

class NeuralNetwork(object):
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.innodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.wih=np.random.normal(0.0,pow(hiddennodes,0.5),(hiddennodes,inputnodes))
        self.who=np.random.normal(0.0,pow(outputnodes,0.5),(outputnodes,hiddennodes))
        self.lr=learningrate
        self.activation_func=sigmoid
    
    def train(self,inputs_list,targets_list):
        inputs=np.array(inputs_list,ndmin=2).T
        targets=np.array(targets_list,ndmin=2).T
        hidden_inputs=np.dot(self.wih,inputs)
        hidden_outputs=self.activation_func(hidden_inputs)
        final_inputs=np.dot(self.who,hidden_outputs)
        final_outputs=self.activation_func(final_inputs)
        errors=targets-final_outputs
        hidden_errors=np.dot(self.who.T,errors)
        self.who+=self.lr*np.dot(
            (errors*final_outputs*(1.0-final_outputs)),
            np.transpose(hidden_outputs)
        )
        self.wih+=self.lr*np.dot(
            (hidden_errors*hidden_outputs*(1.0-hidden_outputs)),
            np.transpose(inputs)
        )
    
    def query(self,input_list):
        inputs=np.array(input_list,ndmin=2).T
        hidden_inputs=np.dot(self.wih,inputs)
        hidden_outputs=self.activation_func(hidden_inputs)
        final_inputs=np.dot(self.who,hidden_outputs)
        final_outputs=self.activation_func(final_inputs)
        return final_outputs
