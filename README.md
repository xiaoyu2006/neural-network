# Neural Network
My first neural network with very simple things!
# Usage
Create it
```
>>> network=NeuralNetwork(inputnodes,hiddennodes,outputnodes,learningrate)
```
Train it
```
# train dataset=[...]
# labels=[...]
>>> network.train(train_dataset,labels)
```
See the result
```
# feature=[...]
>>> network.query(feature)
```
Also change the learningrate by using ```network.lr=newlearningrate```
## Application
A small example of MNIST.
## Tip
You can use ```pickle``` to save the neural network.