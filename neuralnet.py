import numpy 
import scipy.special 
#import matplotlib.pyplot 
#%matplotlib inline
import os 
import scipy.ndimage 



class neuralNetwork : 
	
	def __init__(self,inputnodes, hiddennodes, outputnodes, learningrate) : 
		#Setting nodes in each layer 
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		
		#Link weight matrices, link of node i to node j 
		self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes,self.inodes))
		self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes,self.hnodes))

		self.lr = learningrate

 		#Sigmoid function --> Activatin function
		self.activation_function = lambda x : scipy.special.expit(x)

		pass 
		

	def train(self, inputs_list, targets_list) :
		#Convert lists to 2D arrays 
		inputs = numpy.array(inputs_list, ndmin = 2).T
		targets = numpy.array(targets_list, ndmin =2).T

		#calculate signals into hidden layers 
		hidden_inputs = numpy.dot(self.wih, inputs)
		#Calculate signals coming from hidden layer 
		hidden_outputs = self.activation_function(hidden_inputs)
		#Calculate signals into final output 
		final_inputs = numpy.dot(self.who, hidden_outputs)
		#Calculate final output 
		final_outputs = self.activation_function(final_inputs)

		#Calculating ERROR 
		#Output errors 
		output_errors = targets - final_outputs 
		#Back-propagated errors from hidden layer nodes 
		hidden_errors = numpy.dot(self.who.T, output_errors)

		#Weight Update/refinement for training 
		#Hidden//Output weights 
		self.who += self.lr * numpy.dot((output_errors*final_outputs*(1.0-final_outputs)), numpy.transpose(hidden_outputs))
		#Input//Hidden weights
		self.wih += self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)), numpy.transpose(inputs))

		pass 


	def query(self, inputs_list) : 
		#Convert inputs list to 2D array 
		inputs = numpy.array(inputs_list, ndmin=2).T 

		#Calculate Signals into hidden layer 
		hidden_inputs = numpy.dot(self.wih, inputs)
		#Calculate signals coming from hidden layer 
		hidden_outputs = self.activation_function(hidden_inputs)
		#Calculate signals into hidden layer 
		final_inputs = numpy.dot(self.who, hidden_outputs)
		#Calculate signals coming from final output 
		final_outputs = self.activation_function(final_inputs)


		return final_outputs

#Notification code, apple script 
def notify(title, text):
    os.system("""
              osascript -e 'display notification "{}" with title "{}"'
              """.format(text, title))
    print('\a''\a''\a''\a')
    pass 


#Configuring nodes, learningrate and instance for query 
inputnodes = 784
hiddennodes = 200
outputnodes = 10

learningrate = 0.01 
epochs = 5 

n = neuralNetwork(inputnodes,hiddennodes, outputnodes, learningrate)

#Obtaining Training data 
trainpathname = input('Training File Pathname : ')
#trainpathname = "/Users/AlexisBaudron/Desktop/NeuralNet/mnist_train.csv"
training_data_file = open(trainpathname, 'r')
training_data_list = training_data_file.readlines() 
training_data_file.close()
#Obtaining test data 
testpathname = input('Testing File Pathname : ')
#testpathname = "/Users/AlexisBaudron/Desktop/NeuralNet/mnist_test.csv"
testing_data_file = open(testpathname , 'r')
testing_data_list = testing_data_file.readlines()  
training_data_file.close()  

#Looping through each picture, initiliazizing training data and running the code an epoch amount of times
for e in range(epochs) : 
	for record in training_data_list : 
		all_values = record.split(',') 
		inputs = (numpy.asfarray(all_values[1:]) / 255 * 0.99) + 0.01 
		targets = numpy.zeros(outputnodes) + 0.01
		targets[int(all_values[0])] = 0.99 
		n.train(inputs,targets)
		pass 
	#Creating rotated variations
	#Rotated anticlockwise by 10 degrees 
	inputs_plus10_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval = 0.01,reshape=False)
	n.train(inputs_plus10_img.reshape(784), targets)
	#Rotated anticockswise by 10 degrees 
	inputs_minus10_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval = 0.01,reshape=False)
	n.train(inputs_minus10_img.reshape(784), targets) 
pass 
notify("NeuralNet","Training is finished YAY") 



#Looping through test data and making scorecard 
scorecard = [] 
for record in testing_data_list : 
	all_values = record.split(',') 
	correct_label = int(all_values[0])
	print( "Correct label is : " , correct_label)
	inputs = (numpy.asfarray(all_values[1:])/ 255 * 0.99) + 0.01 
	outputs = n.query(inputs) #query the network with formatted input data 
	label = numpy.argmax(outputs)
	print("Network's answer : " , label)
	if (label==correct_label) : 
		scorecard.append(1) #1 for correct 
	else : 
		scorecard.append(0) #0 for incorrect 
		pass 
	pass 

#Obtaning performance score 
performance_array = numpy.asfarray(scorecard)
performance = performance_array.sum() / performance_array.size
print("Performance : ", performance)


















