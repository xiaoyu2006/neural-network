from network import *

n=NeuralNetwork(784,500,10,0.1)

f=open('mnist-data/mnist_train.csv')
data_list=f.readlines()
f.close()

# TRAIN
epochs=1
for _ in range(epochs):
    print "The %dth epochs."%(_+1)
    count=0
    for record in data_list:
        count+=1
        if count%1000==0:
            print "\tLearning the %dth data..."%count
        all_values=record.strip().split(',')
        inputs=np.asfarray(all_values[1:])/255.0*0.99+0.01
        targets=np.zeros(10)+0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)

del record,data_list,all_values,inputs,targets,count

# SAVE
import pickle
f=open('MNIST.neuralnetwork','w')
pickle.dump(n,f)
f.close()
del pickle

# TEST
right=0
s=0
f=open("mnist-data/mnist_test.csv")
data_list=f.readlines()
f.close()
del f
for record in data_list:
    all_values=record.strip().split(',')
    correct_label=int(all_values[0])
    inputs=np.asfarray(all_values[1:])/255.0*0.99+0.01
    outputs=n.query(inputs)
    label=np.argmax(outputs)
    if label==correct_label:
        right+=1
    s+=1
print "performance =",float(right)/float(s)