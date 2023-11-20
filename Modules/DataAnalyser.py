import os
import pickle
import re
import numpy as np

#Sklearn
from sklearn.feature_extraction import DictVectorizer
from category_encoders import HashingEncoder, TargetEncoder, QuantileEncoder, PolynomialEncoder, HelmertEncoder, BackwardDifferenceEncoder, BinaryEncoder, OneHotEncoder, OrdinalEncoder

#Different Classifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

from Modules.Utils import DictToList, CompleteNop, FormatInstruction, FindFilesExtract
from sklearn.base import clone

#Classifiers created by the Analyzer
Encoders = [
    OrdinalEncoder(handle_unknown= 'value', handle_missing = "value", return_df=False),
    #OneHotEncoder(handle_unknown= 'value', handle_missing = "value", return_df=False),
    #HashingEncoder( hash_method='sha256', return_df=False),
    #HashingEncoder( hash_method='sha256', return_df=False, n_components=32),
    #TargetEncoder(handle_unknown= 'value', handle_missing = "value", return_df=False)
    #QuantileEncoder(handle_unknown= 'value', handle_missing = "value", return_df=False),
    #PolynomialEncoder(handle_unknown= 'value', handle_missing = "value", return_df=False)
    #BinaryEncoder(handle_unknown= 'value', handle_missing = "value", return_df=False)
]

EncoderNames = [
    "OrdinalEncoder",
    #"OneHotEncoder",
    #"HashingEncoder",
    #"HashingEncoder2",
    #"TargetEncoder",
    #"QuantileEncoder",
    #"PolynomialEncoder"
    #Binary",
]

#Classifiers created by the Analyzer
Classifiers = [
    #RandomForestClassifier(criterion='gini', n_estimators =800),
    #KNeighborsClassifier(n_neighbors=6),
    MLPClassifier(alpha=0.0001, max_iter=1000, hidden_layer_sizes=(200, 200, 200, 200, 200), random_state=42, solver='adam'),
    #AdaBoostClassifier(),
]

#Names Associated with the classifiers
Names = [
    #"RandomForest",
    #"KNeighborsClassifier",
    "NeuralNet",
    #"AdaBoost",
]



"""
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
"""
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

#Convert DATA into the sklearn format
def ConvertDataFrequency(Type, Data):
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
            if Tempo in Dictionnary:
                Index = Dictionnary.index(Tempo)
                Result[Index][0]+=1
            if Tempo not in Dictionnary:
                Dictionnary.append(Tempo)
                Result.append([1, 0])
    
    for File in DictionnaryBad:
        for Gadget in File:
            Tempo = {}
            for IndexInstruction in range(len(Gadget)):
                NewInstruction = FormatInstruction(Gadget[IndexInstruction])
                Tempo["Instruction_" + str(len(Gadget) - IndexInstruction - 1)] = NewInstruction;
            if Tempo in Dictionnary:
                Index = Dictionnary.index(Tempo)
                Result[Index][1]+=1
            if Tempo not in Dictionnary:
                Dictionnary.append(Tempo)
                Result.append([0, 1])

    Dictionnary = CompleteNop(Dictionnary)  

    y = []
    for i in Result:
        y.append(i[0]/(i[0] + i[1]))

    return Dictionnary, y

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

    for j in range(len(Encoders)):
        MyEncoder = clone(Encoders[j])
        yt = np.array(y)
        CXt = MyEncoder.fit_transform(X, yt)

        print("Number of Gadgets to train: " + str(len(CXt)))

        count1 = 0
        count2 = 0
        for R in yt:
            if(R == 0):
                count1+=1
            if(R == 1):
                count2+=1
        print(str(count1) + "labeled 0")
        print(str(count2) + "labeled 1")

        #For all classifiers
        for i in range(len(Classifiers)):
            try:
                #Try to fit
                print(Names[i] +' Classifier creation')
                Classifiers[i].fit(CXt, yt)
                SaveClassifier(Classifiers[i], MyEncoder, PathClassifier + Names[i], PathVectorizer + Names[i])
                print(Names[i] +' Saved')
            except Exception as e:
                #If not possible write an error and try the next one
                print('ERROR(AnalyseDataClassifier): Impossible to save classifier ' + Names[i] )
                print(e)