import sys

#Ow modules
from Modules.DataBuilder import BuildData
from Modules.DataAnalyser import AnalyseDataClassifier
from Modules.DataTests import FilesROC, GadgetsROC, GadgetsROCSize, ConfusionMatrix, GadgetsTSNE, GadgetsTreeExtractor
from Modules.DataTool import SyntaxicML, SymbolicAngropML,SymbolicAngrop

ListNameFileToAnalyze = ["notepad.exe"]
Coef = 0.01 #0.003
Thread = 1
FolderBinaryFile = "Data/Files/"
FolderGadgets = "Data/Gadgets/"
FolderClassifiers = "Data/Classifiers/"
FolderFilesToTest = "Data/TestDataSet/"
FolderAnalyzedFiles= "Data/FilesToBeTested/"
NameFileToAnalyze= "notepad.exe"
FolderAnalyzedFiles= "Data/Test/"

ClassifierName = "RopClassifierRandomForest.pkl"
VektorizerName = "RopEncoderRandomForest.ecd"

#Import Different Created functions
print("-----------")
print("Rop tool ML tool")
print("-----------")
print("Pierre-François MAILLARD\n\n")
print("---------------------------------------------")
print("Options:")
print()
print("DATA CREATION")
print("     1- Analyse DATA from sources")
print("CLASSIFIER CREATION")
print("     2- Create ML Classifiers")
print("     3- Create ML Regressors")
print("CLASSIFIER ANALYSIS")
print("     4- Draw ROC from ML Classifier (File Dataset)")
print("     5- Draw ROC from ML Classifier (Gadget Dataset)")
print("     6- Draw ROC from ML Classifier with SMOTE (Gadget Dataset)")
print("     7- Draw ROCs from ML Classifier with gadget sizes (Gadget Dataset)")
print("     8- Matrix Confusion and Accuracy")
print("     9- Compare Angrop Symbolic and ML Angrop on a File")
print("TOOL EXECUTION")
print("     10- Analyse a file with syntaxic ML")
print("     11- Analyse a file with Angrop ML")
print("     12- Analyse a file with ML Classification")
print("     13- Export a Random Forest tree to verify it")
print("---------------------------------------------")

#Entry
if len(sys.argv) > 1:
    A = sys.argv[1]
else: 
    A = input()

if not A.isdigit():
    print("ERROR: Not a Digit")
    exit(0)    
    
if(A == "1"):
    BuildData(FolderBinaryFile, FolderGadgets, 2)
    
if(A == "2"):
    AnalyseDataClassifier(FolderGadgets,FolderClassifiers + "RopClassifier", FolderClassifiers + "RopEncoder")

if(A == "3"):
    print("DELETED")

if(A == "4"):
    print(ClassifierName)
    print(VektorizerName)
    FilesROC(FolderFilesToTest, FolderClassifiers + ClassifierName, FolderClassifiers +  VektorizerName )

if(A == "5"):
   GadgetsROC(FolderGadgets)

if(A == "6"):
   GadgetsROC(FolderGadgets, SmoteToken = 1)

if(A == "7"):
   GadgetsROCSize(FolderGadgets)

if(A == "8"):
    ConfusionMatrix(FolderGadgets, Coef)

if(A == "9"):
    for NameFileToAnalyze in ListNameFileToAnalyze:
        try:
            print(NameFileToAnalyze)
            print("ML ANGROP 0.003:")
            SymbolicAngropML(FolderAnalyzedFiles + NameFileToAnalyze, FolderClassifiers + ClassifierName , FolderClassifiers + VektorizerName, Thread, 0.003)
            print("Classical ANGROP:")
            SymbolicAngrop(FolderAnalyzedFiles + NameFileToAnalyze,Thread)
        except Exception as e:
            print(e)


if(A == "10"):
   SyntaxicML(FolderAnalyzedFiles  + NameFileToAnalyze, FolderClassifiers + ClassifierName ,FolderClassifiers + VektorizerName, Coef)

if(A == "11"):
    SymbolicAngropML(FolderAnalyzedFiles + NameFileToAnalyze, FolderClassifiers + ClassifierName ,FolderClassifiers + VektorizerName, Thread, Coef)

if(A == "12"):
   print("DELETED")

if(A == "13"):
   GadgetsTreeExtractor(FolderGadgets)
