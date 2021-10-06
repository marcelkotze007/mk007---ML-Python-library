import numpy as np
import pandas as pd

class proccessing_data(object):
    def get_data(self, filename):
        dataframe = pd.read_csv(filename)
        #Want to see whats in the file dataframe.head()
        dataframe = dataframe.values        #Converts the dataframe to a nd.array
        X = dataframe[:, :-1]               #Extracts all but the last column from the dataframe
        Y = dataframe[:, -1]

        N = len(X)                         #Gets the number of iterations
        D = len(X.T)                       #Gets the number of features
        #Could also use:
        #N, D = X.shape

        #Must first convert data to a normal distribution 
        for i in range(1, 2, 1):
            #Convert to a normal distribution
            X[:, i] = (X[:,i] - X[:,i].mean())/ X[:,i].std()
        
        
        #Create a new X, for the time_of_day column as it has four category values (one-hot encoding)
        X2 = np.zeros((N, D+3)) #As there are going to be 3 extra features,representing 1
        X2[:, 0 : (D-1)] = X[:, 0 : (D-1)] #The first colums of the vector will be the same, only the last column will
        #split into 4 colums, where it is currently only one column
        
        """
        #Now comes the one-hot encoding for the other 4 colums we are using to replace the last column
        for n in range(N):
            t = int(X[n, D-1])   #loops trough each row of the final column and saves the value as t
            X2[n, t + (D-1)] = 1 #creates a new column and sets the value in that column for that row equal to 1
        """
        #There is a second method that can be used just to elimante the for loop:
        Z = np.zeros((N, 4))  #Create a new array with all 0 values for each of the 4 new features
        Z[np.arange(N), X[:, D-1].astype(np.int32)] = 1 #Firstly arranges the values from 0-3
        #Secondly sets each row equal to 1 and then splits the rows into 4 columns
        X2[:, -4:] = Z  #adds the one-hot encoding to the end of the array
        #assert(np.abs(X2[:, -4:] - Z).sum() < 10e-10) #Just showcases that the values are indeed close to 0
                
        return X2, Y

    def get_binary_data(self, filename):
        """
        extracts the binary data from the dataset
        """
        X, Y = PD.get_data(filename)
        X2 = X[Y <= 1]
        Y2 = Y[Y <= 1]

        return X2, Y2

#filename = "C:/Users/Marcel/OneDrive/Python Courses/Deep Learning/Commerce Project/ecommerce_data.csv"
if __name__ == "__main__":
    PD = proccessing_data()
#X2, Y = PD.get_data(filename)
#X2, Y2 = PD.get_binary_data()
