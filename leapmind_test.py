import numpy as np
from numpy import genfromtxt
from scipy.misc import imread,imsave
import time
import csv,gzip,os

#     layer_0  w0 layer_1  w1  layer_2
#		___             		___
#      -   -                   -   -
#	   -   -==      ___     ===-   -
#		---   ==   -   -  ==    ---
#		___     ===-   -==       |
#      -   -  == =  ---  =       |
#	   -   -==  =        =       |
#		---    =    ___   =      |
#		___   =    -   -  =      |
#      -   - = =   -   -  =      |
#      -   -=  =    ---    =     |
#		---    =     |     =   10|
#        |    =    20|     =     |
#        |    =      |      =    |
#     784|    =     ___     =    |
#        |   =     -   -    =   ___
#		___  =     -   -     = -   -
#      -   - =      ---       =-   -
#	   -   -=                   ---
#		---

def sigmoid(x,derivation=False):
	if(derivation==True):
	    return x*(1-x)
	return 1/(1+np.exp(-x))

#--------------------------mnist extract---------------------------------------
IMAGE_SIZE = 28

def extract_data(filename, num_images):
  #Extract the images into a 4D tensor [image index, y, x, channels].
  #Values are rescaled from [0, 255] down to [-0.5, 0.5].
  
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    #data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data

def extract_labels(filename, num_images):
  #Extract the labels into a vector of int64 label IDs.
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels

train_data_filename = 'train-images-idx3-ubyte.gz'
train_labels_filename = 'train-labels-idx1-ubyte.gz'
test_data_filename = 't10k-images-idx3-ubyte.gz'
test_labels_filename = 't10k-labels-idx1-ubyte.gz'

# Extract it into np arrays.
train_data = extract_data(train_data_filename, 60000)
train_labels = extract_labels(train_labels_filename, 60000)
test_data = extract_data(test_data_filename, 10000)
test_labels = extract_labels(test_labels_filename, 10000)

if not os.path.isdir("mnist/train-images"):
   os.makedirs("mnist/train-images")

if not os.path.isdir("mnist/test-images"):
   os.makedirs("mnist/test-images")
   
ori = np.array([1,0,0,0,0,0,0,0,0,0])

# process train data
with open("mnist/train-labels.csv", 'wb') as csvFile:
  writer = csv.writer(csvFile, delimiter=',', quotechar='"')
  for i in range(len(train_data)):
  	imsave("mnist/train-images/" + str(i) + ".jpg", train_data[i][:,:,0])
	shifted = np.roll(ori, train_labels[i])
	writer.writerow(shifted)


# repeat for test data
with open("mnist/test-labels.csv", 'wb') as csvFile:
  writer = csv.writer(csvFile, delimiter=',', quotechar='"')
  for i in range(len(test_data)):
  	imsave("mnist/test-images/" + str(i) + ".jpg", test_data[i][:,:,0])
  	shifted = np.roll(ori, test_labels[i])
  	writer.writerow(shifted)

#------------------------end of mnist extract-----------------------------------


#-----------------------------training------------------------------------------
train_num = 60000
#load train label
y = genfromtxt('mnist/train-labels.csv', delimiter=',')


np.random.seed(1)

# random weight initialize
w0 = np.random.uniform(-1,1,size = (784,20))
w1 = np.random.uniform(-1,1,size = (20,10))
start = time.time()

#j = epoch
for j in xrange(100):
    for x in xrange(train_num):
		# read image
	    filename = "mnist/train-images/"+str(x)+".jpg"
	    X = imread(filename)
	    X= X.reshape([1,28*28])
	    #binarize image
	    X_low = X<=150
	    X[X_low] = 0
	    X_high = X>150
	    X[X_high] = 1
	    layer_0 = X
	    
	    #forward propagation
	    layer_1 = sigmoid(np.dot(layer_0,w0))
	    layer_2 = sigmoid(np.dot(layer_1,w1))
	    
	    layer_2_error = y[x] - layer_2
	    
	    #backpropagation
	    layer_2_delta = layer_2_error*sigmoid(layer_2,derivation=True)
	    
	    layer_1_error = layer_2_delta.dot(w1.T)
	    
	    layer_1_delta = layer_1_error * sigmoid(layer_1,derivation=True)
	    
	    #weight update
	    w1 += np.matrix(layer_1).T.dot(np.matrix(layer_2_delta))
	    w0 += np.matrix(layer_0).T.dot(np.matrix(layer_1_delta))

#end of training
end = time.time() - start

print "Time taken(train): "+str(end)

#save model
np.savetxt("mnist/w0.csv",w0,delimiter=",")
np.savetxt("mnist/w1.csv",w1,delimiter=",")
np.savetxt("mnist/layer_0.csv",layer_0,delimiter=",")
np.savetxt("mnist/layer_1.csv",layer_1,delimiter=",")
#---------------------------------end of training-------------------------------


#---------------------------------------test------------------------------------
count_correct = 0
count_wrong = 0
test_num = 10000

#read file
w0 = genfromtxt("mnist/w0.csv",delimiter=",")
w1 = genfromtxt("mnist/w1.csv",delimiter=",")
layer_1 = genfromtxt("mnist/layer_1.csv",delimiter=",")
label = genfromtxt("mnist/test-labels.csv",delimiter=",")

start = time.time()

#open test image
for h in xrange(test_num):
	
	#read input image
    filename = "mnist/test-images/"+str(h)+".jpg"
    X = imread(filename)
    X= X.reshape([1,28*28])
    
    #binarize input
    X_low = X<=150
    X[X_low] = 0
    X_high = X>150
    X[X_high] = 1
    
    #forward propagation
    layer_0 = X
    labels = label[h]
    layer_1 = sigmoid(np.dot(layer_0,w0))
    layer_2 = sigmoid(np.dot(layer_1,w1))
    
    #binarize output for comparison with label
    layer_2_low = layer_2<0.9
    layer_2[layer_2_low] = 0
    layer_2_high = layer_2>=0.9
    layer_2[layer_2_high] = 1
    h+=1
    
    #result accuracy count
    if (layer_2 == labels).all():
        count_correct += 1
    else:
        count_wrong += 1
        
time = time.time() - start
print "Time taken(test): " + str(time) + "s"
print "Correct Recognize: " + str(count_correct)
print "Wrong Recognize: " + str(count_wrong)
print "Accuracy: " + str((((float(count_correct))/test_num) * 100))

#-----------------------------------end of test---------------------------------
