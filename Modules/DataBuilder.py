import angr, MyAngrop.angrop
from shutil import copyfile

import os
import warnings
import logging

import multiprocessing 
import multiprocessing.pool
import time

from random import randint

"""
Classes which allow to create a multithread pool in a pool. It was not possible du to python condition.
"""
class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass
        
class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess
    
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)

"""
All function which allow to search create ROP chains with specific conditions. The usefuls gadgets will be saved
"""   
def FindAllGadgetsRax(p, File):
    ListGadgets = []
    MyGadgets = []
    
    TempoGadgetFile = File + "_BadRax.txt"
    print(TempoGadgetFile)
    copyfile(File, TempoGadgetFile)

    #set RAX
    while True:

        #On charge les gadgets possibles
        MyRop = p.analyses.ROP()
        MyRop.load_gadgets(TempoGadgetFile)
        MyGadgets = MyRop._gadgets.copy()
        #print(MyRop._gadgets)
        #On essaie de former une chain
        try:
            chain = MyRop.set_regs(rax=0x1337)
        except:
            break
    
        #On supprime tout les gadgets present dans la chaine de notre liste
        for CurentChainPart in chain._gadgets:
            for CurrentGadget in MyGadgets:
                if CurrentGadget.addr == CurentChainPart.addr:
                    ListGadgets.append(CurrentGadget) 
                    MyGadgets.remove(CurrentGadget)     
                    break
        
        #On ecrase l'ancien fichier de gadget avec un nouveau sans rien
        MyRop._gadgets = set(MyGadgets)
        MyRop.save_gadgets(TempoGadgetFile)
    
    #On a une liste de gadgets non-utilisable et une liste de gadgets utilisables     
    print("utilisables: " + str(len(ListGadgets)) + " inutilisables: " + str(len(MyGadgets)))
    
    os.remove(TempoGadgetFile)
    
    return ListGadgets, MyGadgets

def FindAllGadgetsRbx(p, File):
    ListGadgets = []
    MyGadgets = []
    
    TempoGadgetFile = File + "_BadRbx.txt"
    
    copyfile(File, TempoGadgetFile)
    
    #set RAX
    while True:

        #On charge les gadgets possibles
        MyRop = p.analyses.ROP()
        MyRop.load_gadgets(TempoGadgetFile)
        MyGadgets = MyRop._gadgets.copy()
        
        #On essaie de former une chain
        try:
            chain = MyRop.set_regs(rbx=0x1337)
        except:
            break
    
        #On supprime tout les gadgets present dans la chaine de notre liste
        for CurentChainPart in chain._gadgets:
            for CurrentGadget in MyGadgets:
                if CurrentGadget.addr == CurentChainPart.addr:
                    ListGadgets.append(CurrentGadget) 
                    MyGadgets.remove(CurrentGadget)     
                    break
        
        #On ecrase l'ancien fichier de gadget avec un nouveau sans rien
        MyRop._gadgets = set(MyGadgets)
        MyRop.save_gadgets(TempoGadgetFile)
    
    #On a une liste de gadgets non-utilisable et une liste de gadgets utilisables     
    print("utilisables: " + str(len(ListGadgets)) + " inutilisables: " + str(len(MyGadgets)))
    
    os.remove(TempoGadgetFile)
    
    return ListGadgets, MyGadgets

def FindAllGadgetsRcx(p, File):
    ListGadgets = []
    MyGadgets = []
    
    TempoGadgetFile = File + "_BadRcx.txt"
   
    copyfile(File, TempoGadgetFile)
    
    #set RAX
    while True:

        #On charge les gadgets possibles
        MyRop = p.analyses.ROP()
        MyRop.load_gadgets(TempoGadgetFile)
        MyGadgets = MyRop._gadgets.copy()
        
        #On essaie de former une chain
        try:
            chain = MyRop.set_regs(rcx=0x1337)
        except:
            break
    
        #On supprime tout les gadgets present dans la chaine de notre liste
        for CurentChainPart in chain._gadgets:
            for CurrentGadget in MyGadgets:
                if CurrentGadget.addr == CurentChainPart.addr:
                    ListGadgets.append(CurrentGadget) 
                    MyGadgets.remove(CurrentGadget)     
                    break
        
        #On ecrase l'ancien fichier de gadget avec un nouveau sans rien
        MyRop._gadgets = set(MyGadgets)
        MyRop.save_gadgets(TempoGadgetFile)
    
    #On a une liste de gadgets non-utilisable et une liste de gadgets utilisables     
    print("utilisables: " + str(len(ListGadgets)) + " inutilisables: " + str(len(MyGadgets)))
    
    os.remove(TempoGadgetFile)
    
    return ListGadgets, MyGadgets

def FindAllGadgetsRdx(p, File):
    ListGadgets = []
    MyGadgets = []
    
    TempoGadgetFile = File + "_BadRdx.txt"
    
    copyfile(File, TempoGadgetFile)
    
    #set RAX
    while True:

        #On charge les gadgets possibles
        MyRop = p.analyses.ROP()
        MyRop.load_gadgets(TempoGadgetFile)
        MyGadgets = MyRop._gadgets.copy()
        
        #On essaie de former une chain
        try:
            chain = MyRop.set_regs(rdx=0x1337)
        except:
            break
    
        #On supprime tout les gadgets present dans la chaine de notre liste
        for CurentChainPart in chain._gadgets:
            for CurrentGadget in MyGadgets:
                if CurrentGadget.addr == CurentChainPart.addr:
                    ListGadgets.append(CurrentGadget) 
                    MyGadgets.remove(CurrentGadget)     
                    break
        
        #On ecrase l'ancien fichier de gadget avec un nouveau sans rien
        MyRop._gadgets = set(MyGadgets)
        MyRop.save_gadgets(TempoGadgetFile)
    
    #On a une liste de gadgets non-utilisable et une liste de gadgets utilisables     
    print("utilisables: " + str(len(ListGadgets)) + " inutilisables: " + str(len(MyGadgets)))
    
    os.remove(TempoGadgetFile)
    
    return ListGadgets, MyGadgets

def FindAllGadgetsWriteMem(p, File):
    ListGadgets = []
    MyGadgets = []
    
    TempoGadgetFile = File + "_BadWMeme.txt"
    
    copyfile(File, TempoGadgetFile)
    
    #set RAX
    while True:

        #On charge les gadgets possibles
        MyRop = p.analyses.ROP()
        MyRop.load_gadgets(TempoGadgetFile)
        MyGadgets = MyRop._gadgets.copy()
        
        #On essaie de former une chain
        try:
            chain = MyRop.write_to_mem(0x61b100, b"/bin/sh\0")
        except:
            break
    
        #On supprime tout les gadgets present dans la chaine de notre liste
        for CurentChainPart in chain._gadgets:
            for CurrentGadget in MyGadgets:
                if CurrentGadget.addr == CurentChainPart.addr:
                    ListGadgets.append(CurrentGadget) 
                    MyGadgets.remove(CurrentGadget)     
                    break
        
        #On ecrase l'ancien fichier de gadget avec un nouveau sans rien
        MyRop._gadgets = set(MyGadgets)
        MyRop.save_gadgets(TempoGadgetFile)
    
    #On a une liste de gadgets non-utilisable et une liste de gadgets utilisables     
    print("utilisables: " + str(len(ListGadgets)) + " inutilisables: " + str(len(MyGadgets)))
    
    os.remove(TempoGadgetFile)
    
    return ListGadgets, MyGadgets

def FindAllGadgetsAddMem(p, File):
    ListGadgets = []
    MyGadgets = []
    
    TempoGadgetFile = File + "_BadAMem.txt"
    
    copyfile(File, TempoGadgetFile)
    
    #set RAX
    while True:

        #On charge les gadgets possibles
        MyRop = p.analyses.ROP()
        MyRop.load_gadgets(TempoGadgetFile)
        MyGadgets = MyRop._gadgets.copy()
        
        #On essaie de former une chain
        try:
            chain = MyRop.add_to_mem(0x804f124, 0x41414141)
        except:
            break
    
        #On supprime tout les gadgets present dans la chaine de notre liste
        for CurentChainPart in chain._gadgets:
            for CurrentGadget in MyGadgets:
                if CurrentGadget.addr == CurentChainPart.addr:
                    ListGadgets.append(CurrentGadget) 
                    MyGadgets.remove(CurrentGadget)     
                    break
        
        #On ecrase l'ancien fichier de gadget avec un nouveau sans rien
        MyRop._gadgets = set(MyGadgets)
        MyRop.save_gadgets(TempoGadgetFile)
    
    #On a une liste de gadgets non-utilisable et une liste de gadgets utilisables     
    print("utilisables: " + str(len(ListGadgets)) + " inutilisables: " + str(len(MyGadgets)))
    
    os.remove(TempoGadgetFile)
    
    return ListGadgets, MyGadgets

"""
Functions which allow to write Result (Good and Bad gadgets) in a file
"""
def WriteResult(Path, File,Good, Bad):
    with open(Path + File, 'w') as f:
        #Write good gadgets
        for item in Good:
            f.write("T %s\n" % hex(item.addr))
            f.write(item.block)
            f.write("\n")
        #Write bad gadgets
        for item in Bad:
            f.write("F %s\n" % hex(item.addr))
            f.write(item.block)
            f.write("\n")

"""
Main Function which find the different useful gadgets for a File. This function is multithreaded.
"""
def AnalyseFile(MyData):

    #Different argument from the function
    PathFiles= MyData[0]
    PathResult= MyData[1]
    File= MyData[2]
    ThreadNumbers= MyData[3]

    print( File  + " analysed")

    #Name of the file which will saved the gadgets DATA 
    TempoGadgetName = "Gadgets" + File + ".txt"

    #Find The files to analyse
    p = angr.Project(PathFiles + File)
    rop = p.analyses.ROP()

    #Find all gadgets and save them (symbolic states)
    rop.find_gadgets(processes=ThreadNumbers)
    if(len(rop.gadgets) == 0):
        return

    rop.save_gadgets(TempoGadgetName)

    #Etat initial
    print("Nombre de gadgets courant : " + str(len(rop.gadgets)))

    #We will save all the gadgets used to take control of a register
    try:
        Good, Bad = FindAllGadgetsRax(p,TempoGadgetName)
        WriteResult(PathResult, File+"_Rax.txt", Good, Bad)
        """
        Good, Bad = FindAllGadgetsRbx(p,TempoGadgetName)
        WriteResult(PathResult, File+"_Rbx.txt", Good, Bad)
        Good, Bad = FindAllGadgetsRcx(p,TempoGadgetName)
        WriteResult(PathResult, File+"_Rcx.txt", Good, Bad)
        Good, Bad = FindAllGadgetsRdx(p,TempoGadgetName)
        WriteResult(PathResult, File+"_Rdx.txt", Good, Bad)
        Good, Bad = FindAllGadgetsWriteMem(p,TempoGadgetName)
        WriteResult(PathResult, File+"_Mem.txt", Good, Bad)
        Good, Bad = FindAllGadgetsAddMem(p,TempoGadgetName)
        WriteResult(PathResult, File+"_Add.txt", Good, Bad)
        """
        print("Success " + File)
    except:
        print("Error " + File)

    #Remove the temporary file 
    os.remove(TempoGadgetName)

"""
MAIN: From Files, fidn all gadgets and analyse it to create useful gadgets, Multithread on each CPU
"""
def BuildData(PathSourceFiles,PathResult, ThreadNumbers):
    #Set logs to minimum
    logging.getLogger('angr').setLevel('CRITICAL')
    logging.getLogger('angrop').setLevel('CRITICAL')
    
    #List all files from Directory
    PathFiles = PathSourceFiles
    ListFiles = os.listdir(PathFiles)

    #Sorted all files to work on the shorter first
    TempoFile = []
    for File in ListFiles:
        TempoFile.append([File, os.path.getsize(PathFiles + File)])
        
    TempoFile = sorted(TempoFile, key=lambda TempoFile: TempoFile[1])     
        
    ListFiles = []
    for File in TempoFile:
        ListFiles.append(File[0])    

    #Create the Data for each thread
    i=0
    MyData = []
    for file in ListFiles:
        MyData.append([PathFiles,PathResult, file, ThreadNumbers])
        i+=1

    #Create a pool of thread (Angrop create a pool too, so it is necessary to create a pool without daemon)
    with NestablePool() as pool:
        pool.map(AnalyseFile, MyData)
    pool.close()
    pool.join()

