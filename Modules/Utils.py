import re
import os

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

#Find all files from a directory and all its subdirectory 
def FindFiles(PathFiles):     
    FilesGadgets = {} 
    FilesGadgets[PathFiles] = ParseFile(PathFiles)
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
        if(i[0] + i[1] > 3):
            y.append(i[0]/(i[0] + i[1]))
        else:
            if i[0] >= 1:
                y.append(0.1)
            else:
                y.append(0)


    return Dictionnary, y
