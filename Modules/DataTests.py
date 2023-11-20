import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydot

from sklearn.base import clone

from sklearn.model_selection import train_test_split
#from sklearn.preprocessing  import OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn import metrics

from sklearn.tree import export_graphviz

from category_encoders import HashingEncoder, TargetEncoder, QuantileEncoder, PolynomialEncoder, HelmertEncoder, BackwardDifferenceEncoder, BinaryEncoder, OneHotEncoder, OrdinalEncoder

from Modules.Utils import DictToList, CompleteNop, FormatInstruction, FindFilesExtract, ConvertData, ConvertDataFrequency

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE
import seaborn as sns

#Classifiers created by the Analyzer
Encoders = [
    #OrdinalEncoder(handle_unknown= 'value', handle_missing = "value", return_df=False),
    #OneHotEncoder(handle_unknown= 'value', handle_missing = "value", return_df=False),
    #HashingEncoder( hash_method='sha256', return_df=False),
    #HashingEncoder( hash_method='sha256', return_df=False, n_components=32),
    TargetEncoder(handle_unknown= 'value', handle_missing = "value", return_df=False),
    #PolynomialEncoder(handle_unknown= 'value', handle_missing = "value", return_df=False),
    #HelmertEncoder(handle_unknown= 'value', handle_missing = "value", return_df=False),
    #BackwardDifferenceEncoder(handle_unknown= 'value', handle_missing = "value", return_df=False, drop_invariant=True, verbose=3),
    #BinaryEncoder(handle_unknown= 'value', handle_missing = "value", return_df=False)
    #QuantileEncoder(handle_unknown= 'value', handle_missing = "value", return_df=False),
]

EncoderNames = [
    #"OrdinalEncoder",
    #"OneHotEncoder",
    #"HashingEncoder",
    #"HashingEncoder2",
    "TargetEncoder",
    #"PolynomialEncoder",
    #"HelmertEncoder",
    #"BackawardEncoder",
    #"Binary",
    #"QuantileEncoder",
]

#Classifiers created by the Analyzer
Classifiers = [
    #RandomForestClassifier(criterion='gini', n_estimators =800),
    RandomForestClassifier(criterion='entropy', n_estimators =800, class_weight= None),
    #RandomForestClassifier(criterion='entropy', n_estimators =800, class_weight= 'balanced'),
    #KNeighborsClassifier(n_neighbors=5),
    #AdaBoostClassifier(),
    #MLPClassifier(alpha=0.0001, max_iter=1000, hidden_layer_sizes=(200, 200, 200, 200, 200), random_state=42, solver='adam'),
]

#Names Associated with the classifiers
Names = [
    #"RandomForestGini",
    "RandomForestEntropyUnbalanced",
    #"RandomForestEntropyBalanced",
    #"KNeighborsClassifier",
    #"AdaBoost",
    #"NeuralNet_200_5",
]

def MyOversample(X, y):
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    return X,y

#Function to draw a ROC curve
def DrawRoc(FPR, TPR, Name):
	#Fabrication d'une courbe ROC
	plt.title('Receiver Operating Characteristic - AUC:' + str(metrics.auc(FPR, TPR)))
	plt.plot(FPR, TPR, color ="red")
	plt.legend(loc = 'lower right')
	plt.plot([0, 1.01], [-0.01, 1.01],'r--')
	plt.xlim([0, 1.01])
	plt.ylim([0, 1.01])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig(Name + '.png')
	plt.close()

#Save ROC with the list of Y and Yest
def ROCsaveSizeGadget(X, Yest, Y, Name):
    for SizeGadget in range(len(X[0])-2):
        SizeGadget+=2

        LTPR = []
        LFPR = []
        Iterateur = np.arange(0, 1, float(1/1000))
        for Scale in Iterateur: 
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for i in range(len(Yest)):
                Size = len(X[i].tolist()) - X[i].tolist().count("nop")
                if(Size == SizeGadget):
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
        DrawRoc(LFPR, LTPR, Name+ "_Size" + str(SizeGadget)) 

#Save ROC with the list of Y and Yest
def ROCsave(Yest, Y, Name):
    LTPR = []
    LFPR = []

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
        print( str(Scale)+"=> TP: " + str(TP) + ", TN: " + str(TN) + ", FN: " + str(FN) + ", FP: " + str(FP) + ", LTPR: "+ str(TP/(TP+FN)) + ", LFPR: "+ str(FP/(FP+TN)))

    DrawRoc(LFPR, LTPR, Name) 

"""
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
"""
# MAIN function to test classifier on File
def FilesROC(PathTested, PathClassifier,PathVectorizer):
    LTPR = []
    LFPR = []

    Data = FindFilesExtract(PathTested)

    #Load classifier
    with open(PathClassifier, 'rb') as f:
        MyClassifier = pickle.load(f)
    with open(PathVectorizer, 'rb') as f:
        vectorizer = pickle.load(f)

    #print(len(vectorizer.feature_names_out_))

    X, y = ConvertData("Rax", Data)
    Xt = CompleteNop(X, MaxSize = 11)   
    Xt = DictToList(Xt)

    print("Number of Gadgets to Test: " + str(len(Xt)))

    #Transorm the DATA with vectorizer
    Xt = vectorizer.transform(Xt)

    #Try Classifier
    Yresult = MyClassifier.predict_proba(Xt)

    #AssociateTab.sort(key=lambda x: x[0], reverse=False) 
    ROCsave(Yresult, y, "ResultFile")

# MAIN function to test classifier on Gadget
def GadgetsROC(PathGadgets, SmoteToken = 0):
    Data = FindFilesExtract(PathGadgets)
    X, y = ConvertData("Rax", Data)
    Xt = CompleteNop(X)    
    CXt = DictToList(Xt)

    for j in range(len(Encoders)):
        MyEncoder = clone(Encoders[j])
        yt = np.array(y)
        try:
            print(EncoderNames[j])
            Xt = MyEncoder.fit_transform(CXt, yt)
        except Exception as e:
            print('ERROR(AnalyseDataClassifier): Impossible to encode ' + EncoderNames[j])
            print(e)  

        for i in range(len(Classifiers)):
            try:
                kf =  KFold(n_splits=5, random_state=42, shuffle=True)
                Nb = 0
                SYresult = []
                STest = []
                for train_index, test_index in kf.split(Xt):
                    print("Index : " + str(Nb))
                    Nb+=1
                    X_train, X_test = Xt[train_index], Xt[test_index]
                    y_train, y_test = yt[train_index], yt[test_index]
                    print(Names[i] +' Classifier creation')
                    MyClassifier = clone(Classifiers[i])
                    MyClassifier.fit(X_train, y_train)
                    if SmoteToken == 1:
                        X_train, y_train = MyOversample(X_train, y_train)
                    Yresult = MyClassifier.predict_proba(X_test)
                    SYresult.extend(Yresult)
                    STest.extend(y_test)
                if SmoteToken == 1:
                    ROCsave(SYresult, STest, "ResultGadgetsSMOTE" + Names[i] + EncoderNames[j])
                else:
                    ROCsave(SYresult, STest, "ResultGadgets" + Names[i] + EncoderNames[j])

            except Exception as e:
                print('ERROR(AnalyseDataClassifier): Impossible to fit classifier ' + Names[i] + EncoderNames[j] + str(Nb))
                print(e)  

# MAIN function to test classifier on Gadget
def GadgetsTSNE(PathGadgets):

    Data = FindFilesExtract(PathGadgets)
    X, y = ConvertData("Rax", Data)
    Xt = CompleteNop(X)    
    CXt = DictToList(Xt)

    for j in range(len(Encoders)):
        MyEncoder = clone(Encoders[j])
        yt = np.array(y)
        Xt = MyEncoder.fit_transform(CXt, yt)

        Xt, yt = MyOversample(Xt, yt)

        Xt, X_test, yt, y_test = train_test_split(Xt, yt, test_size=0.95, random_state=42)

        X_R = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(Xt)

        tsne_result_df = pd.DataFrame({'tsne_1': X_R[:,0], 'tsne_2': X_R[:,1], 'label': yt})
        fig, ax = plt.subplots(1)
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=15)
        lim = (X_R.min()-5, X_R.max()+5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

        fig.savefig('DataRepresentation' + EncoderNames[j] + '.png')

#Main function to get some accuracy
def ConfusionMatrix(PathGadgets, Beta):
    Data = FindFilesExtract(PathGadgets)
    X, y = ConvertData("Rax", Data)
    Xt = CompleteNop(X)    
    CXt = DictToList(Xt)

    #X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.33, random_state=42)
    #print(type(X_train))

    for j in range(len(Encoders)):
        MyEncoder = clone(Encoders[j])
        yt = np.array(y)
        Xt = MyEncoder.fit_transform(CXt, yt)

        for i in range(len(Classifiers)):
            SPrecision1 = 0
            SPrecision2 = 0
            SRecall1 = 0
            SRecall2 = 0
            SF1 =0
            SF2 =0
            try:
                kf =  KFold(n_splits=5, random_state=42, shuffle=True)
                Nb = 0
                for train_index, test_index in kf.split(Xt):
                    print("Index : " + str(Nb))
                    Nb+=1
                    X_train, X_test = Xt[train_index], Xt[test_index]
                    y_train, y_test = yt[train_index], yt[test_index]

                    print("Number of Gadgets to train: " + str(len(X_train)))
                    print("Number of Gadgets to test: " + str(len(X_test)))

                    print(Names[i] +' Classifier creation')
                    MyClassifier = clone(Classifiers[i])
                    MyClassifier.fit(X_train, y_train)
                    Yresult = MyClassifier.predict_proba(X_test)
                    print("Classfier Created"  + Names[i] + EncoderNames[j])
                    TP = 0
                    FP = 0
                    FN = 0
                    TN = 0
                    for j in range(len(Yresult)):
                        if (y_test[j] == 1) and (Yresult[j][1] >= Beta):
                            TP+=1
                        if (y_test[j] == 0) and (Yresult[j][1] >= Beta):
                            FP+=1
                        if (y_test[j] == 0) and (Yresult[j][1] < Beta):
                            TN+=1
                        if (y_test[j] == 1) and (Yresult[j][1] < Beta):
                            FN+=1    
                    print()
                    print("Accuracy: " + str((TP+TN)/(TP+FN+TN+FN)))
                    print() 
                    Precision1 = TP/(TP+FP)
                    Recall1 = TP/(TP+FN)
                    print("Precision score (Good): " + str(Precision1))
                    print("Recall score (Good): " + str(Recall1))
                    F1 = 2 * (Precision1 * Recall1) / (Precision1 + Recall1)
                    print("F1 score: " + str(F1))
                    print()
                    Precision2 = TN/(TN+FN)
                    Recall2 = TN/(TN+FP)
                    print("Precision score (Bad): " + str(Precision2))
                    print("Recall score (Bad): " + str(Recall2))
                    F2 = 2 * (Precision2 * Recall2) / (Precision2 + Recall2)
                    print("F2 score: " + str(F2))
                    print()
                    print("Average F = " + str((F1 + F2)/2))
                    print()
                    print("Confusion Matrix")
                    print("            Predicted")
                    print("Actual  ( " + str(TP) + " " + str(FP) + " )") 
                    print("        ( " + str(FN) + " " + str(TN) + " )") 
                    print()
                    SPrecision1 += Precision1
                    SPrecision2 += Precision2
                    SRecall1 += Recall1
                    SRecall2 += Recall2
                    SF1 += F1
                    SF2 += F2

                SPrecision1 /= 5
                SPrecision2 /= 5
                SRecall1 /= 5
                SRecall2 /= 5
                SF1 /= 5
                SF2 /= 5

                print("Average Precision score (Good): " + str(SPrecision1))
                print("Average Recall score (Good): " + str(SRecall1))
                print("Average F1 score: " + str(SF1))
                print()
                print("Average Precision score (Bad): " + str(SPrecision2))
                print("Average Recall score (Bad): " + str(SRecall2))
                print("Average F2 score: " + str(SF2))
                print()
                print("Average F = " + str((SF1 + SF2)/2))

            except Exception as e:
                print('ERROR(AnalyseDataClassifier): Impossible to fit classifier ' + Names[i] + EncoderNames[j])
                print(e)  

# MAIN function to test classifier on Files
def GadgetsROCSize(PathGadgets):
    Data = FindFilesExtract(PathGadgets)
    X, y = ConvertData("Rax", Data)
    Xt = CompleteNop(X)    
    CXt = DictToList(Xt)
    NCXt = np.array(CXt)

    for j in range(len(Encoders)):
        MyEncoder = clone(Encoders[j])
        yt = np.array(y)
        Xt = MyEncoder.fit_transform(CXt, yt)

        for i in range(len(Classifiers)):
            try:
                kf =  KFold(n_splits=5, random_state=42, shuffle=True)
                SYresult = []
                STest = []
                SX_test = []
                for train_index, test_index in kf.split(Xt):
                    X_train, X_test = Xt[train_index], Xt[test_index]
                    y_train, y_test = yt[train_index], yt[test_index]
                    OldX = NCXt[test_index]
                    print(Names[i] +' Classifier creation')
                    MyClassifier = clone(Classifiers[i])
                    MyClassifier.fit(X_train, y_train)
                    Yresult = MyClassifier.predict_proba(X_test)
                    SYresult.extend(Yresult)
                    STest.extend(y_test)
                    SX_test.extend(OldX)
                ROCsaveSizeGadget(SX_test,SYresult, STest, "ResultGadgets" + Names[i] + EncoderNames[j])
                
            except Exception as e:
                print('ERROR(AnalyseDataClassifier): Impossible to fit classifier ' + Names[i] )
                print(e)  

#Main function to get a random tree
def GadgetsTreeExtractor(PathGadgets):
    Data = FindFilesExtract(PathGadgets)
    X, y = ConvertData("Rax", Data)
    Xt = CompleteNop(X)    
    CXt = DictToList(Xt)

    Classifier = RandomForestClassifier(criterion='entropy', n_estimators =800)

    for j in range(len(Encoders)):
        MyEncoder = clone(Encoders[j])
        yt = np.array(y)
        CXt = MyEncoder.fit_transform(CXt, yt)

        MyClassifier = clone(Classifier)
        MyClassifier.fit(CXt, yt)
        estimator = MyClassifier.estimators_[5]
        export_graphviz(estimator, out_file='Tree' + EncoderNames[j] + '.dot',rounded = True, proportion = False, precision = 2, filled = True)

        (graph,) = pydot.graph_from_dot_file('Tree' + EncoderNames[j] + '.dot')
        graph.write_png('Tree' + EncoderNames[j] + '.png')
        print( 'Tree' + EncoderNames[j] + '.png created')
    
    return


