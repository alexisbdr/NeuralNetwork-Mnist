#Opening and plotting image from mnist data file 
filePath = input("File Pathname : ")
data_file = open(filePath,'r')
data_list = data_file.readlines() 
data_file.close() 
import numpy 
import matplotlib.pyplot
%matplotlib inline 
all_values = data_list[0].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array , cmap='Greys' , interpolation='None')
