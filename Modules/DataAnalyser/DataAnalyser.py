import os
import pickle
import re
import numpy

#Sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing  import OrdinalEncoder

#Different Classifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

""""
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
"""

#Classifiers created by the Analyzer
Classifiers = [
    #AdaBoostClassifier(),
    RandomForestClassifier(criterion='gini', n_estimators =400),
    #KNeighborsClassifier(n_neighbors=10),
    #MLPClassifier(alpha=0.0001, max_iter=1000),
]

#Names Associated with the classifiers
Names = [
    #"AdaBoost",
    "RandomForest",
    #"KNeighborsClassifier",
    #"NeuralNet",
]

#Convert a Dict of Data into a List for classification
def DictToList(X):
    NewX = []
    for Gadget in X:
        NewGadget = []
        for i in sorted(Gadget):
            NewGadget.append(Gadget[i])
        NewX.append(NewGadget)
    return NewX

#With the MaxLen de X, complete gadgets with Nop instruction
def CompleteNop(X, MaxSize = -1):
    Maxlen  = 0

    if(MaxSize == -1):
        for i in range(len(X)):
            if(len(X[i]) > Maxlen):
                Maxlen = len(X[i])
    else:
        Maxlen = MaxSize

    for j in range(len(X)):
        for i in range(Maxlen):
            if not "Instruction_" + str(i) in X[j]:
                 X[j]["Instruction_" + str(i)] = "nop"

    return X

#Format the instruction to replace addr into 0xXX 
def FormatInstruction(Instruction):
    NewInstruction = re.sub(r"\b0[xX]([0-9a-fA-F]+)\b","0xXX", Instruction)
    return NewInstruction

#Parse line format from angrop
def GoodFormat(Line):
    if ":\t" in Line:
        Line = Line.split(":\t")[1]
    Line = Line.replace("\t", " ")
    Line = Line.replace(" \n", "")
    Line = Line.replace("\n", "")
    return Line

#Read Data from a Gadget file with label
def ParseFile(Path, Binary, File):
    TotalName = Path + Binary + "_" + File + ".txt"
    i=0
    G = []
    List = {"Good": [], "Bad": []}
    
    #Read the File with gadget
    fichier = open(TotalName, "r")
    for Line in fichier:
        if G != [] and (Line[0] == "T" or Line[0] == "F"):
            if Line[0] == "T":
                List["Good"].append(G)
            if Line[0] == "F":
                List["Bad"].append(G)    
                
        if Line[0] == "T":
            Current = "T"
            G = []      
        elif Line[0] == "F":
            Current = "F"
            G = []
        else: 
            #Modification of the line with a good format
            Line = GoodFormat(Line)
            G.append(Line) 
    
    fichier.close()
    
    return List

#Find all files from a directory and all its subdirectory 
def FindFilesExtract(PathFiles):
    FilesGadgets = {}
    
    ListFiles = os.listdir(PathFiles)

    #Lister les fichiers existants et les analyses
    for file in ListFiles:
        Name = file.rpartition("_")[0]
        SubFile = file.rpartition("_")[2].split(".")[0]
        if Name not in FilesGadgets:
            FilesGadgets[Name] = {}
            FilesGadgets[Name][SubFile] = {}
        else:
            FilesGadgets[Name][SubFile] = {}
            
    for Binary in FilesGadgets:
        for File in FilesGadgets[Binary]:
            FilesGadgets[Binary][File] = ParseFile(PathFiles, Binary,File)
            
    return FilesGadgets

#Convert DATA into the sklearn format
def ConvertData(Type, Data):
    Dictionnary = []
    Result = []
    DictionnaryGood= []
    DictionnaryBad = []

    Tempo = {}
    
    #Find all of the Datas from the files
    for i in Data:
        DictionnaryGood.append(Data[i][Type]["Good"])
        DictionnaryBad.append(Data[i][Type]["Bad"])

    #Create our dictionnary with the gadgets
    for File in DictionnaryGood:
        for Gadget in File:
            Tempo = {}
            for IndexInstruction in range(len(Gadget)):
                NewInstruction = FormatInstruction(Gadget[IndexInstruction])
                Tempo["Instruction_" + str(len(Gadget) - IndexInstruction - 1)] = NewInstruction;
            if Tempo not in Dictionnary:
                Dictionnary.append(Tempo)
                Result.append(1)
    
    for File in DictionnaryBad:
        for Gadget in File:
            Tempo = {}
            for IndexInstruction in range(len(Gadget)):
                NewInstruction = FormatInstruction(Gadget[IndexInstruction])
                Tempo["Instruction_" + str(len(Gadget) - IndexInstruction - 1)] = NewInstruction;
            if Tempo not in Dictionnary:
                Dictionnary.append(Tempo)
                Result.append(0)

    Dictionnary = CompleteNop(Dictionnary)  
        
    return Dictionnary, Result

#Final function to save a classifier 
def SaveClassifier(Classifier,vectorizer, PathClassifier, PathVectorizer):
    with open(PathClassifier + ".pkl", 'wb') as f:
        pickle.dump(Classifier, f)   

    with open(PathVectorizer + ".ecd", 'wb') as f:
        pickle.dump(vectorizer, f)       

#Classifier analyser
def AnalyseDataClassifier(SourcesPath, PathClassifier, PathVectorizer):
    Data = FindFilesExtract(SourcesPath)
     
    #Rax tool with the right DATA
    X, y = ConvertData("Rax", Data)
    X = CompleteNop(X)    
    X = DictToList(X)

    #Convert our instructions into Discrete Values
    MyEncoder = OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value=-1)
    X = MyEncoder.fit_transform(X)

    #For all classifiers
    for i in range(len(Classifiers)):
        try:
            #Try to fit
            print(Names[i] +' Classifier creation')
            Classifiers[i].fit(X, y)
            SaveClassifier(Classifiers[i], MyEncoder, PathClassifier + Names[i], PathVectorizer + Names[i])
            print(Names[i] +' Saved')
        except Exception as e:
            #If not possible write an error and try the next one
            print('ERROR(AnalyseDataClassifier): Impossible to save classifier ' + Names[i] )
            print(e)

