import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Just for the moore_class import:
import re #Used to remove certain charecters from imported data

class one_D_Linear:
    def get_data(self, filename):
        """
        Loads the data.
        Takes the filename as an argument.
        Returns all of the input in array X
        Returns the output as an array Y
        """
        #Load the data using numpy
        dataset = pd.read_csv(filename).values
        plt.scatter(dataset[:,0], dataset[:,1])
        #plt.show()
        X = dataset[:,0]
        Y = dataset[:,1]
        return X, Y

    #Loading the data into a list first and then converting it into a numpy array
    def long_method(self):
        #Loading in the data
        X = []
        Y = []
        for line in open("C:/Users/Marcel/OneDrive/Python Courses/Deep Learning/Linear Regression/Data/data_1d.csv"):
            x, y = line.split(',')
            X.append(float(x))
            Y.append(float(y))
        
        #Next convert the data into a numpy array:
        X = np.array(X)
        Y = np.array(Y)

        #Plot the data:
        plt.scatter(X, Y)
        #plt.show()
    
    #Coding the calculations we did in the notes
    def calculate_a_b(self, X, Y):
        """
        Calculates the a and b values for a linear equation of any 1-D dataset.
        Takes in Two arguments and produces a estimated value (Yhat) which is then graphed, which creates a line of
        best fit.
        X = X-values
        Y = Y-values
        """
        denominator = X.dot(X) - X.mean() * X.sum()
        #Numerator
        a = (X.dot(Y) - Y.mean() * X.sum()) / denominator #Check to make sure it is X.sum(), and not X.mean()
        b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

        #Calculates the predicted Y
        Yhat = a*X + b

        #Plot the graph
        plt.scatter(X, Y)
        plt.plot(X, Yhat)
        plt.show()

        return Yhat, a, b

    def R_square(self, Y, Yhat):
        """
        Calculates the R squared that indicates how well the model fits the data.
        The closer R squared is to 1, the better the fit. 
        Must first calculate the expected values of Y i.e. Yhat, using calculate_a_b() function
        """
        dif1 = Y - Yhat
        dif2 = Y - Y.mean()
        
        #Rater use the dot function as it multiplies and sums the values, vector multiplication,
        #as 100x1 * 1x100 = a single value, the sum                                    
        R = 1 - dif1.dot(dif2)/dif2.dot(dif2) 
        print("The value of R squared is", R)
        return R

class moore_law:
    def get_data(self):
        X = []
        Y = []
        #The re function is used her
        non_decimal = re.compile(r'[^\d]+')
        
        for line in open("C:/Users/Marcel/OneDrive/Python Courses/Deep Learning/Linear Regression/Data/moore.csv"):
            r = line.split('\t')
            
            #Adds only the year and the number of transistors to the data
            x = int(non_decimal.sub('', r[2].split('[')[0]))
            y = int(non_decimal.sub('', r[1].split('[')[0]))

            #Adds the values to the list
            X.append(x)
            Y.append(y)
        
        #Convert lists into np arrays
        X = np.array(X)
        Y = np.array(Y)

        #Plot the data:
        plt.scatter(X, Y)
        plt.show()

        Y = np.log(Y)
        plt.scatter(X, Y)
        plt.title("Distribution using the log of Y")
        plt.show()

        return X, Y
    
    def Time_double(self, a, b):
        """
        Calculates the time it takes for transitor count to double. Takes in arguments of a and b. 
        Produces a value in years.
        """
        #Calculating how long it takes for the transitor count to double:
        # log(tc) = a*year + b         As difined by the linear regression model
        # tc = exp(a*year) * exp(b)    As difined by our expression before refining it to a straight line, using log
        # 2*tc = 2 * exp(a*year) * exp(b)
        #      = exp(ln(2)) * exp(a*year) * exp(b)
        #      = exp(ln(2) + a*year) * exp(b)
        # exp(a*year2) * exp(b) = exp(a*year1 + ln(2)) * exp(b)
        # a*year2 = a*year1 + ln2
        # year2 = year1 + ln(2)/a
        b = None
        time_to_double = np.log(2)/a
        print("Time to double: %s years" %time_to_double)

#filename = "C:/Users/Marcel/OneDrive/Python Courses/Deep Learning/Linear Regression/Data/data_1d.csv"

"""
#Calls the class:
imp_data = get_data()
#imports the data and stores it in X and Y respectively
X, Y = imp_data.short_method()
#imp_data.long_method()
#Creates the line of best fitting by calculating a and b
Yhat = imp_data.calculate_a_b(X, Y)
print("The result is ", imp_data.R_square(Y, Yhat))
"""
"""
#Moore's law functions
ML = moore_law()
X, Y = ML.get_data()
#Using functions calculated in previous class
imp_data = One_D_Linear()
Yhat, a, b = imp_data.calculate_a_b(X, Y)
imp_data.R_square(Y, Yhat)
ML.Time_double(a, b)
"""