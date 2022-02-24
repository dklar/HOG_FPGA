import os
import cv2 as cv
from random import randrange
'''
This program calls a executable file that marks postive and negative 
objects (humans) in a database. If you think the list is long enough
kill this python script because it will call every picture in the 
database.
'''
hog_capture_path = "/home/dennis/Schreibtisch/HOG_FPGA/HOG_Capture/build/HOG_Capture "

def validFile(filename: str) -> bool:
    """Validate file type"""
    if  filename.lower().endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".bmp"): 
        return True
    else:
        return False

def createPositiveList(directory):
    '''
    Mark positive objects on database
    '''
    print("Mark positive objects...")
    for filename in os.listdir(directory):
        if validFile(filename): 
            arg0 = hog_capture_path
            arg1 = os.path.join(directory, filename)+" "
            #arg2 = "/home/dennis/Schreibtisch/HOG_train_Aug/HOG_Capture/positive.txt"
            arg2 = "/home/dennis/Downloads/GRAZ/positive.txt"
            callarguments = arg0+arg1+arg2
            os.system(callarguments)
            continue
        else:
            continue

def createNegativeList(directory):
    '''
    Mark negative objects on database
    '''
    print("Mark negative objects...")
    for filename in os.listdir(directory):
        if validFile(filename): 
            arg0 = hog_capture_path
            arg1 = os.path.join(directory, filename)+" "
            #arg2 = "/home/dennis/Schreibtisch/HOG_train_Aug/HOG_Capture/negative.txt"
            arg2 = "/home/dennis/Downloads/GRAZ/negative.txt"
            callarguments = arg0+arg1+arg2
            os.system(callarguments)
            continue
        else:
            continue

def megaNegativeList(directory):
    '''
    Generate Random areas with no detectable objects and adds them on the negative list
    '''
    print("Mark negative objects...")
    f = open("negative.txt","w")
    for filename in os.listdir(directory):
        if validFile(filename):
            picture = cv.imread(directory + filename)
            for i in range(10):
                y = randrange(0,picture.shape[0]-128)
                x = randrange(0,picture.shape[1]-64)
                f.write(directory+filename+"|"+str(x)+"|"+str(y)+"|64|128\n")
    

createPositiveList("/home/dennis/Databases/HOG_database/Graz02_personen/")
#createPositiveList("/home/dennis/Downloads/person/")
#createNegativeList("/home/dennis/Downloads/none/")
#megaNegativeList("/home/dennis/Downloads/GRAZ/Graz02_none/")
