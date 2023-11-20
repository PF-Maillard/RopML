import angr
import pickle
import time
from colorama import Fore, Back, Style

from Modules.Utils import DictToList, CompleteNop, FormatInstruction, FindFilesExtract, GoodFormat

def  DisplaySyntaxic(Result, Coef):
    #Display Result
    A = 0
    for Gadget, Value in Result:
        Color = Fore.WHITE
        if Value >= Coef:
            Color = Fore.RED
            if Value >= (1-Coef)*(1/4) + Coef:
                Color = Fore.YELLOW
            if Value >= (1-Coef)*(1/2) + Coef:
                Color = Fore.GREEN

            print(Fore.BLUE + "Gadget Value:" + Fore.BLACK + Color + " " + str(Value))
            for Number, Instruction in Gadget.items():
                if(Instruction != "nop"):
                    print(Fore.WHITE + "=> " + Instruction)
            A +=1
            print("\n")

    print(Back.BLUE +"Useful Gadgets: " + str(A) + " on " + str(len(Result)) + "(Coef=" + str(Coef)+ ")" + Back.BLACK + "\n")

    return

#Find gadgets from a specific file through Angrop   
def GetGadgetsFromFile(SourcesPath):
    MyGadgets = []
    p = angr.Project(SourcesPath)
    rop = p.analyses.ROP()

    MyGadgets = rop.find_gadgets_without_analysis()
    return MyGadgets

#Function to test a file through a list of gadget and a classifier
def ClassifyGadgets(X, PathClassifier,PathVectorizer ):
    #Load classifier
    with open(PathClassifier, 'rb') as f:
        MyClassifier = pickle.load(f)
    with open(PathVectorizer, 'rb') as f:
        vectorizer = pickle.load(f)

    Xt = CompleteNop(X, MaxSize = len(vectorizer.feature_names_out_))           
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
def SyntaxicML(SourcesPath, PathClassifier, PathVectorizer, Coef):
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

    Result = ClassifyGadgets(X, PathClassifier,PathVectorizer)
    
    DisplaySyntaxic(Result, Coef)

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

