import PIL.Image as image
from network import *

import pickle
f=open('MNIST.neuralnetwork')
n=pickle.load(f)

imgname=raw_input('Handwriting image name? (Should be in the same folder) ')
img=image.open(imgname).convert('1').resize((28,28))

img_array=1-(numpy.array(img.getdata())/255.0*0.99+0.01)
print 'Image is',img_array

print "The network thinks it's the number %d"%numpy.argmax(n.query(img_array))