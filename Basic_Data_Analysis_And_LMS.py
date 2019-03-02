# -*- coding: utf-8 -*-
""""************************************************
**    Autors: Sebastian_Laverde_Alfonso 119414    **
**    Name: Machine_Learning_Assignment_1         **
*************************************************"""


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def read_cvs(file):
    
    """Reads a .cvs file"""
    
    data_set = pd.read_csv(file)
    columns = data_set.columns.values.tolist()
    
    return (data_set, columns)

def petal_vs_sepal_lenght_by_species():
    
    """Select, Filter and Plot the data from a pandas DataFrame globally created called 'df'"""
    
    select_df = df.iloc[:, [0, 2, 4]]
    setosa_df = select_df[(select_df.species == 'setosa')]
    virginica_df = select_df[(select_df.species == 'virginica')]
    versicolor_df = select_df[(select_df.species == 'versicolor')]
    
    x_set = setosa_df.iloc[:,1]
    y_set = setosa_df.iloc[:,0]

    x_vir = virginica_df.iloc[:,1]
    y_vir = virginica_df.iloc[:,0]

    x_ver = versicolor_df.iloc[:,1]
    y_ver = versicolor_df.iloc[:,0]

    #%matplotlib inline

    fig = plt.figure()
    plot = fig.add_subplot(111)
    
    plt.title("Petal lenght Vs Sepal Lenght")
    
    plt.xlabel(columns[2])
    plt.ylabel(columns[0])
    
    yellow_patch = mpatches.Patch(color='y', label='Setosa')
    blue_patch = mpatches.Patch(color='b', label='Virginica')
    red_patch = mpatches.Patch(color='r', label='Versicolor')
    
    plt.legend(handles=[yellow_patch, blue_patch, red_patch])
    
    plot.scatter(x_set, y_set, marker = '^', color = 'y', s = 50)
    plot.scatter(x_vir, y_vir, marker = '^', color = 'b', s = 50)
    plot.scatter(x_ver, y_ver, marker = '^', color = 'r', s = 50)
   

def petal_vs_sepal_lenght():
    
    df, columns = read_cvs("IrisDataSet.csv")
    select = df.iloc[:, [0, 2]]
    x = select.iloc[:,1]
    y = select.iloc[:,0]
    
    fig = plt.figure()
    plot = fig.add_subplot(111)
    
    plt.title("Petal lenght Vs Sepal Lenght")
    plt.xlabel(columns[2])
    plt.ylabel(columns[0])
    plot.scatter(x, y, marker = '^', color = 'orange', s = 50)

def species_features_description(species):
    print("------------------------\n\n", species,"description: \n")
    for feature in range(0,4):
        data_set, columns = read_cvs("IrisDataSet.csv")
        attribute=data_set.iloc[:,[feature,4]]
        attribute_by_species=attribute[(attribute.species == species)]
        statistics=attribute_by_species.describe()
        print(statistics, "\n")
        
species_features_description("setosa")
species_features_description("virginica")
species_features_description("versicolor")
   

df, columns = read_cvs("IrisDataSet.csv")
petal_vs_sepal_lenght_by_species()
petal_vs_sepal_lenght()
dataset=df.loc[:,['sepal length','species']]
dataset=dataset[(dataset.species=='setosa') | (dataset.species=='virginica')]
dataset=dataset.values[1:]

# THE LMS ALGORITHM

for line in range(len(dataset)):
    if (dataset[line,1]=='setosa'):
        dataset[line,1]=-1.0
    else:
        dataset[line,1]=1.0

def rss (dataset, w):
    one=np.ones_like(dataset[:,0])
    x=np.array([one,dataset[:,0]])
    c=dataset[:,-1]
    y=w.dot(x)
    r=(c-y)**2
    return r.sum()

def lms (dataset, iterations, eta):
    print("------------LMS Algorithm--------------\n")
    np.random.seed(7)
    w=np.random.uniform(low=-1.0,high=1.0,size=2)
    counter = 0
    while counter<iterations:
        rand=np.random.randint(len(dataset))
        x=np.array([1.0,dataset[rand,0]])
        c=dataset[rand,-1]
        y=w.dot(x)
        error=c-y
        w+=(eta*error*x)
        counter+=1
        if counter==iterations:
            print("Last Iteration:  RSS: "+str(rss(dataset, w)))
            print("The weight vector: "+str(w))
    
    x=dataset[:,0]
    y=dataset[:,1]
    x1=np.linspace(4,8,100)
    y1=x1*w[1]+w[0]
    fig=plt.figure()
    plot = fig.add_subplot(111)
    plot.scatter(x,y)
    plot.plot(x1,y1)
    
    return w
w=lms(dataset,1000000,0.0005)
print("\n----------------Task 4b----------------\n\nAnalysing the maximum and minimum (to establish the interpolating class space) and the average (to determine which species have the data more clustered and are more away from the otherspecies to define regions) of input data, we found out that setosa is the easiest to distinguish in every feature but espectially in the petal length; and versicolor is the hardest because it's feature data could also be from the virginica class. Finally sepal and petal lengths are the most important features (Take a look on plot #1)\n\n-------------Plots-------------\n\nPlot1 - clustering the species\nPlot2 - analizing features\nPlot3 - LMS algorithm's result")
