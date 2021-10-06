import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime

"""
#The manual method of loading data
X = []
for line in open("data_2d.csv"):     #Opens the specified file in order to read the file
    row = line.split(',')            #Indicates where each row ends, where a new row begins
    sample = list(map(float, row))   #Indicates to python that the values added are floats, and not strings, which is the default
    X.append(sample)                 #Adds each row to the back of the list

X = np.array(X)                      #Converts the list to an numpy array
print(X)
print(X.shape)                       #Checks the shape of the numpy array, should be (100, 3)
"""
"""
#Using pandas to import data
X = pd.read_csv("data_2d.csv", header = None) #Function reads the data and imports it into python
print(X.head())                               #Returns the first 5 rows of the data
print(X.iloc[0])                              #Both return the specified row
print(X.ix[0])                                #Rather use iloc as it is beter than ix
print(X [X[0] < 5])                           #Returns the rows, where the X[0] value is less than 5
print(X[[0,2]])                               #Returns specific rows 
print(X.info())
#M = X.as_matrix()                            #Converts the dataframe to a numpy array
V = X.values                                  #Converts a dataframe to a numpy array, the new version as such rather use it 
print(V)
"""
"""
df = pd.read_csv("international-airline-passengers.csv", engine="python", skipfooter=3)
df.columns = ["Month", "Passengers"]               #Rename the column titles using a list
df["ones"] = 1                                     #Adds a new column with each row having a value of one
print(df.head())                                   #Shows the top lines in the data                

datetime.strptime("1949-05", "%Y-%m")              #Creates a datetime object
print(type(datetime.strptime("1949-05", "%Y-%m"))) #Verifies the format of the object
dt.datetime(1949, 5, 1, 0, 0)

#Using the apply method:
#df['x1x2'] = df.apply(lambda row: row['x1'] * row['x2'], axis = 1)
#def get_interaction(row):                         #Using a function 
#    return row['x1']*row['x2']
#df['x1x2'] = df.apply(get_interaction, axis = 1)
#interactions = []                    #Using a for loop to iterate trough each row and add values, should never use as it is slow
#for idx, row in df.iterrows():
#    x1x2 = row['x1'] * row['x2']
#    interactions.append(x1x2)
#df['x1x2'] = interactions

def add_pass(row):
    total = row["Passengers"]
    #print(total)
    if total != row["Passengers"]:
        total = total + row["Passengers"]
    return total

df['Datetime'] = df.apply(lambda row: datetime.strptime(row['Month'], "%Y-%m"), axis = 1)
df["Passengers*2"] = df.apply(lambda row: row["Passengers"] + row["Passengers"], axis = 1 )
df["Total Passengers"] = df.apply(add_pass, axis = 1)
print(df.info())                              #Verifies that the datetime column containts the datetime objects

print (df.head())

print(df.iloc[0]) 
""" 
"""
#Joining two tables:
#First import the tables:
t1 = pd.read_csv("table1.csv")
t2 = pd.read_csv("table2.csv")
#print(t1, t2)                       #Verifies that the tables were imported correctly

m = pd.merge(t1, t2, on = "user_id") #The on argument states in which column the two tables should join
t1.merge(t2, on = "user_id")         #If no column is specified using the on argument, the row index will be used
print(m)
"""