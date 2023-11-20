import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import angr
import time

from Modules.DataAnalyser.DataAnalyser import *
from sklearn.model_selection import train_test_split

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

#Function to draw a ROC curve
def DrawRoc(FPR, TPR, Name):
	#Fabrication d'une courbe ROC
	plt.title('Receiver Operating Characteristic')
	plt.plot(FPR, TPR, color ="red")
	plt.legend(loc = 'lower right')
	plt.plot([0, 1.01], [0, 1.01],'r--')
	plt.xlim([0, 1.01])
	plt.ylim([0, 1.01])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig(Name + '.png')
	plt.close()
	
	return;
	
#Format the instruction to replace addr into 0xXX 
def FormatInstruction(Instruction):
    NewInstruction = re.sub(r"\b0[xX]([0-9a-fA-F]+)\b","0xXX", Instruction)
    return NewInstruction

#parse format of a line from angrop 
def GoodFormat(Line):
    if ":\t" in Line:
        Line = Line.split(":\t")[1]
    Line = Line.replace("\t", " ")
    Line = Line.replace(" \n", "")
    Line = Line.replace("\n", "")
    return Line

#Read Data from a Gadget file
def ParseFile(Path):
    TotalName = Path 
    i=0
    G = []
    List = {"Bad": [], "Good": []}
    
    fichier = open(TotalName, "r")
    for Line in fichier:
        if G != [] and ((Line[0] == "T") or (Line[0] == "F")):
            if (Current == "F"):
                List["Bad"].append(G)      
            if (Current == "T"):
                List["Good"].append(G)     
                
        if (Line[0] == "T") or (Line[0] == "F"):
            Current = Line[0]
            G = []      
        else: 
            Line = GoodFormat(Line)
            G.append(Line)   
            
    if (Current == "F"):
        List["Bad"].append(G)      
    if (Current == "T"):
        List["Good"].append(G)             
    
    fichier.close()
    
    return List

#Find all files from a directory and all its subdirectory 
def FindFiles(PathFiles):     
    FilesGadgets = {} 
    FilesGadgets[PathFiles] = ParseFile(PathFiles)
    return FilesGadgets

#Convert DATA into the sklearn format
def ConvertDataTest(Data, Vectorizer):
    Dictionnary = []
    Result = []
    DictionnaryGood= []
    DictionnaryBad = []

    Tempo = {}
    
    #Find all of the Datas from the files
    for i in Data:
        DictionnaryGood.append(Data[i]["Good"])
        DictionnaryBad.append(Data[i]["Bad"])

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

    Dictionnary = CompleteNop(Dictionnary, MaxSize = Vectorizer.n_features_in_) 
    Dictionnary = DictToList(Dictionnary) 

    return Dictionnary, Result

#read the statistic classafier 
def ReadClassifier(PathClassifierStats):
    ClassifierStat = []

    f = open(PathClassifierStats, mode="r")
    for Line in f:
        ObjectInstruction = Line.split(": ")
        ObjectInstruction[1] = float(ObjectInstruction[1])
        ClassifierStat.append(ObjectInstruction)
    f.close()

    return ClassifierStat

# MAIN function ot test the stat DATA
def ROCTestData(PathTested, PathClassifier,PathVectorizer):
    LTPR = []
    LFPR = []
    AUC = 0.8

    Data = FindFilesExtract(PathTested)

    #Load classifier
    with open(PathClassifier, 'rb') as f:
        MyClassifier = pickle.load(f)
    with open(PathVectorizer, 'rb') as f:
        vectorizer = pickle.load(f)

    X, ResultAngrop = ConvertData("Rax", Data)
    X = CompleteNop(X)    
    X = DictToList(X)

    #Transorm the DATA with vectorizer
    Xt = vectorizer.transform(X)

    #Try Classifier
    ClassifierResult = MyClassifier.predict_proba(Xt)
    
    AssociateTab = []
    for i in range(len(X)):
        AssociateTab.append([X[i], ClassifierResult[i][1], ResultAngrop[i]]) 

    AssociateTab.sort(key=lambda x: x[1], reverse=False) 

    Iterateur = np.arange(0, 1, float(1/1000))
    for Scale in Iterateur:
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(len(AssociateTab)):
            if (AssociateTab[i][2] == 1) and (AssociateTab[i][1] >= Scale):
                TP+=1
            if (AssociateTab[i][2] == 0) and (AssociateTab[i][1] >= Scale):
                FP+=1
            if (AssociateTab[i][2] == 0) and (AssociateTab[i][1] < Scale):
                TN+=1
            if (AssociateTab[i][2] == 1) and (AssociateTab[i][1] < Scale):
                FN+=1      
        LTPR.append(TP/(TP+FN))
        LFPR.append(FP/(FP+TN))
        print( str(Scale)+"=> TP: " + str(TP) + ", TN: " + str(TN) + ", FN: " + str(FN) + ", FP: " + str(FP))

    DrawRoc(LFPR, LTPR,"Result")    

#Save ROC with the list of Y and Yest
def ROCsave(Yest, Y, Name):
    LTPR = []
    LFPR = []
    AUC = 0.8

    Iterateur = np.arange(0, 1, float(1/1000))

    for Scale in Iterateur:
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(len(Yest)):
            if (Y[i] == 1) and (Yest[i][1] >= Scale):
                TP+=1
            if (Y[i] == 0) and (Yest[i][1] >= Scale):
                FP+=1
            if (Y[i] == 0) and (Yest[i][1] < Scale):
                TN+=1
            if (Y[i] == 1) and (Yest[i][1] < Scale):
                FN+=1    
        LTPR.append(TP/(TP+FN))
        LFPR.append(FP/(FP+TN))
        print( str(Scale)+"=> TP: " + str(TP) + ", TN: " + str(TN) + ", FN: " + str(FN) + ", FP: " + str(FP))

    DrawRoc(LFPR, LTPR,"ResultROC" +Name) 

#ROC Curve on New Gadget
def ROCTestDataNewGadget(PathGadgets):

    Data = FindFilesExtract(PathGadgets)
    X, y = ConvertData("Rax", Data)
    X = CompleteNop(X)    
    X = DictToList(X)

    MyEncoder = OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value=-1)
    X = MyEncoder.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    for i in range(len(Classifiers)):
        try:
            print(Names[i] +' Classifier creation')
            Classifiers[i].fit(X_train, y_train)
            Yresult = Classifiers[i].predict_proba(X_test)
            ROCsave(Yresult, y_test, Names[i])
        except Exception as e:
            print('ERROR(AnalyseDataClassifier): Impossible to fit classifier ' + Names[i] )
            print(e)   

#Find gadgets from a specific file through Angrop   
def GetGadgetsFromFile(SourcesPath):
    MyGadgets = []
    p = angr.Project(SourcesPath)
    rop = p.analyses.ROP()

    #Not the same number of gadgets due to this constraint

    """
    if not self._satisfies_mem_access_limits(final_state):
    l.debug("... too many symbolic memory accesses")
    return None

    rop.find_gadgets()
    MyGadgets = rop.gadgets.copy()
    """

    MyGadgets = rop.find_gadgets_without_analysis()
    return MyGadgets

#Function to test a file through a list of gadget and a classifier
def TestGadget(X, PathClassifier,PathVectorizer ):
    #Load classifier
    with open(PathClassifier, 'rb') as f:
        MyClassifier = pickle.load(f)
    with open(PathVectorizer, 'rb') as f:
        vectorizer = pickle.load(f)

    Xt = CompleteNop(X, MaxSize = vectorizer.n_features_in_)           
    Xt = DictToList(Xt)

    #Transorm the DATA with vectorizer
    Xt = vectorizer.transform(Xt)

    #Try Classifier
    ClassifierResult = MyClassifier.predict_proba(Xt)
    
    AssociateTab = []
    for i in range(len(X)):
        AssociateTab.append([X[i], ClassifierResult[i][1]]) 

    AssociateTab.sort(key=lambda x: x[1], reverse=False) 

    return AssociateTab

#Main function to Execute to analyse a file
def AnalyseFileGadget(SourcesPath, PathClassifier, PathVectorizer, Coef):
    X = []
    NewX = []

    ListX = GetGadgetsFromFile(SourcesPath)    
    for i in range(len(ListX)):
        ListX[i] = ListX[i].split("\n")
        for j in range(len(ListX[i])):
            ListX[i][j] = GoodFormat(ListX[i][j])

    for Gadget in ListX:
        Tempo = {}
        for IndexInstruction in range(len(Gadget)):
            NewInstruction = FormatInstruction(Gadget[IndexInstruction])
            Tempo["Instruction_" + str(len(Gadget) - IndexInstruction - 1)] = NewInstruction;
        if Tempo not in X:
            X.append(Tempo)

    Result = TestGadget(X, PathClassifier,PathVectorizer)
    
    #Display Result
    for Gadget, Value in Result:
        print("Value from my Gadget: " + str(Value))
        for Number, Instruction in Gadget.items():
            if(Instruction != "nop"):
                print(Instruction)
        print("\n")
    
    A = 0
    for Gadget, Value in Result:
        if Value >= Coef:
            A+=1
    print("Useful Gadgets: " + str(A) + " on " + str(len(Result)) )

#MainAnalyse a file with symbolic and ML 
def SymbolicAngrop(SourcesPath, ThreadNumbers):

    p = angr.Project(SourcesPath)
    rop = p.analyses.ROP()

    #Find all gadgets and save them (symbolic states)
    t1 = time.time()
    rop.find_gadgets(processes = ThreadNumbers)
    try:
        chain = rop.set_regs(rax=0x1337)
        chain.print_payload_code()
    except Exception as e:
        print(e)
        print("ERROR: Build Chain")
        return
    t2 = time.time()
    delta = t2 - t1
    print(f"Analysis speed {delta} seconds")

#MainAnalyse a file with symbolic and ML 
def SymbolicAngropML(SourcesPath,PathClassifier, PathVectorizer, ThreadNumbers, MyCoef):

    p = angr.Project(SourcesPath)
    rop = p.analyses.ROP()

    with open(PathClassifier, 'rb') as f:
        MyClassifier = pickle.load(f)
    with open(PathVectorizer, 'rb') as f:
        MyVectorizer = pickle.load(f)

    #Find all gadgets and save them (symbolic states)
    t1 = time.time()
    rop.find_gadgets_with_ML(MyClassifier, MyVectorizer, Coef = MyCoef, processes = ThreadNumbers)
    try:
        chain = rop.set_regs(rax=0x1337)
        chain.print_payload_code()
    except Exception as e:
        print(e)
        print("ERROR: Build Chain")
        return
    t2 = time.time()
    delta = t2 - t1
    print(f"Analysis speed {delta} seconds")

