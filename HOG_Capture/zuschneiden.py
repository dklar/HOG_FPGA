import cv2
'''
Take the output textfile from execute.py, the marked objects, and cut out 
the objects and save them to separate pictures, with the pre defined const.
size 128x64
'''

def createPictures(positiveFiles,negativeFiles):
    print("Read positive file examples")
    pathToWrite = "/home/dennis/Schreibtisch/HOG_train_Aug/HOG_Capture/Database_new/"

    fpos = open(positiveFiles,'r')
    lines = fpos.readlines()
    for index, line in enumerate(lines):
        line = line.strip()
        #line = line[:-1]
        values = line.split("|")
        image = cv2.imread(values[0],cv2.IMREAD_GRAYSCALE)
        x = int(values[1])
        y = int(values[2])
        w = int(values[3])
        h = int(values[4])
        ROI_image = image[y:y+h,x:x+w]
        ROI_image = cv2.resize(ROI_image,(64,128),interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(pathToWrite+"pos_{}.jpg".format(index),ROI_image)
    fpos.close()

    print("Read negative file examples")
    fneg = open(negativeFiles,'r')
    lines = fneg.readlines()
    for index, line in enumerate(lines):
        line = line.strip()
        #line = line[:-1]
        values = line.split("|")
        image = cv2.imread(values[0],cv2.IMREAD_GRAYSCALE)
        x = int(values[1])
        y = int(values[2])
        w = int(values[3])
        h = int(values[4])
        ROI_image = image[y:y+h,x:x+w]
        ROI_image = cv2.resize(ROI_image,(64,128))
        cv2.imwrite(pathToWrite+"neg_{}.jpg".format(index),ROI_image)

    fneg.close()
    print("Reading data done...")

#positiveFiles = "/home/dennis/Schreibtisch/HOG_train_Aug/HOG_Capture/positive.txt"
#negativeFiles = "/home/dennis/Schreibtisch/HOG_train_Aug/HOG_Capture/negative.txt"
positiveFiles = "/home/dennis/Downloads/GRAZ/positive.txt"
negativeFiles = "/home/dennis/Downloads/GRAZ/negative.txt"

createPictures(positiveFiles,negativeFiles)
