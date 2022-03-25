import numpy as np
import time
import joblib
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_det_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

'''
    Change here the paths not in the functions!
'''
positiveFiles_acc = "/home/dennis/Schreibtisch/HOG_FPGA/HOG_train/build/positive_acc_L1.data"
negativeFiles_acc = "/home/dennis/Schreibtisch/HOG_FPGA/HOG_train/build/negative_acc_L1.data"
positiveFiles_apr = "/home/dennis/Schreibtisch/HOG_FPGA/HOG_train/build/positive_apr_L1.data"
negativeFiles_apr = "/home/dennis/Schreibtisch/HOG_FPGA/HOG_train/build/negative_apr_L1.data"

def trainAndSave(positiveFiles,negativeFiles,fileNameOutput):
    '''
    Trainiert den SVM Klassifizierer und schreibt ihn in:\n
    a) Einer numpy Datei\n
    b) In einer C Datei als C Quellcode\n

    '''
    print("\t\tTrain the SVM \nWith data of the output of HOG application (for HLS)")
    print("Positive file path : " + positiveFiles)
    print("Negative file path : " + negativeFiles)
    print("Read positive file examples")

    data= []
    labels = []

    dataPos = []
    dataNeg = []

    fpos = open(positiveFiles,'r')
    lines = fpos.readlines()
    for index, line in enumerate(lines):
        line = line.strip()
        line = line[:-1]
        values = line.split(";")
        numpyList = np.array(values,dtype=np.float)

        if np.any(np.isnan(numpyList)):
            print("NAN in Pos.. skip example")
            continue
        if (not np.any(np.isfinite(numpyList))):
            print("oo in Pos.. skip example")
            continue
        dataPos.append(numpyList)
        data.append(numpyList)
        labels.append(1)
    fpos.close()
    print("Read negative file examples")
    fneg = open(negativeFiles,'r')
    lines = fneg.readlines()
    for index, line in enumerate(lines):
        line = line.strip()
        line = line[:-1]
        values = line.split(";")
        numpyList = np.array(values,dtype=np.float)

        if np.any(np.isnan(numpyList)):
            #print("NAN in Neg.. skip example")
            continue
        if (not np.any(np.isfinite(numpyList))):
            #print("oo in Neg.. skip example")
            continue
        dataNeg.append(numpyList)
        data.append(numpyList)
        labels.append(0)
    fneg.close()
    print("Reading data done...")


    le = LabelEncoder()
    labels = le.fit_transform(labels)

    print("Constructing training/testing split...")
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), labels, test_size=0.20, random_state=42)
    print("Training Linear SVM classifier...")
    model = LinearSVC(penalty='l1',dual=False)
    start = time.time()
    model.fit(trainData, trainLabels)
    stop = time.time()
    print(f"Training time: {stop - start}s")
    print(" Evaluating classifier on test data ...")
    predictions = model.predict(testData)
    print(classification_report(testLabels, predictions))

    file = open(fileNameOutput+".cpp",'w')
    classifier = model.coef_[0]
    intercept = model.intercept_
    count = 0
    file.write("float Intercept = {};\n".format(intercept))
    file.write("float classifier[] = {\n")
    for val in classifier:
        toWrite = "{:.5f}".format(val)
        file.write(toWrite + ",")
        count+=1
        if count>20:
            file.write("\n")
            count=0
    file.close()

    joblib.dump(model, fileNameOutput+".npy")

def Precision_Recall_curve_plot(testData, testLabels, model):
    y_score = model.decision_function(testData)
    average_precision = average_precision_score(testLabels, y_score)
    disp = plot_precision_recall_curve(model, testData, testLabels)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
    plt.show()

def plotCurves():
    '''
        Plot the Detection error trade off curves for accurate and approximate classifier
    '''
    data_apr = []
    labels_apr = []

    dataPos_apr = []
    dataNeg_apr = []

    data_acc = []
    labels_acc = []

    dataPos_acc = []
    dataNeg_acc = []
    
    negative_file_apr = open(negativeFiles_apr,'r')
    positive_file_apr = open(positiveFiles_apr,'r')

    negative_file_acc = open(negativeFiles_acc,'r')
    positive_file_acc = open(positiveFiles_acc,'r')


    lines = positive_file_apr.readlines()
    for _, line in enumerate(lines):
        line = line.strip()
        line = line[:-1]
        values = line.split(";")
        numpyList = np.array(values,dtype=np.float)

        if np.any(np.isnan(numpyList)):
            print("NAN in Pos.. skip example")
            continue
        if (not np.any(np.isfinite(numpyList))):
            print("oo in Pos.. skip example")
            continue
        dataPos_apr.append(numpyList)
        data_apr.append(numpyList)
        labels_apr.append(1)
    positive_file_apr.close()

    lines = positive_file_acc.readlines()
    for _, line in enumerate(lines):
        line = line.strip()
        line = line[:-1]
        values = line.split(";")
        numpyList = np.array(values,dtype=np.float)

        if np.any(np.isnan(numpyList)):
            print("NAN in Pos.. skip example")
            continue
        if (not np.any(np.isfinite(numpyList))):
            print("oo in Pos.. skip example")
            continue
        dataPos_acc.append(numpyList)
        data_acc.append(numpyList)
        labels_acc.append(1)
    positive_file_acc.close()

    lines = negative_file_acc.readlines()
    for index, line in enumerate(lines):
        line = line.strip()
        line = line[:-1]
        values = line.split(";")
        numpyList = np.array(values,dtype=np.float)

        if np.any(np.isnan(numpyList)):
            print("NAN in Neg.. skip example")
            continue
        if (not np.any(np.isfinite(numpyList))):
            print("oo in Neg.. skip example")
            continue
        dataNeg_acc.append(numpyList)
        data_acc.append(numpyList)
        labels_acc.append(0)
    negative_file_acc.close()
    
    lines = negative_file_apr.readlines()
    for index, line in enumerate(lines):
        line = line.strip()
        line = line[:-1]
        values = line.split(";")
        numpyList = np.array(values,dtype=np.float)

        if np.any(np.isnan(numpyList)):
            print("NAN in Neg.. skip example")
            continue
        if (not np.any(np.isfinite(numpyList))):
            print("oo in Neg.. skip example")
            continue
        dataNeg_apr.append(numpyList)
        data_apr.append(numpyList)
        labels_apr.append(0)
    negative_file_apr.close()



    le_apr = LabelEncoder()
    labels_apr = le_apr.fit_transform(labels_apr)

    le_acc = LabelEncoder()
    labels_acc = le_acc.fit_transform(labels_acc)


    (trainData_acc, testData_acc, trainLabels_acc, testLabels_acc) = train_test_split(np.array(data_acc), labels_acc, test_size=0.20, random_state=42)
    (trainData_apr, testData_apr, trainLabels_apr, testLabels_apr) = train_test_split(np.array(data_apr), labels_apr, test_size=0.20, random_state=42)

    model_acc = LinearSVC()
    model_acc.fit(trainData_acc, trainLabels_acc)

    model_apr = LinearSVC()
    model_apr.fit(trainData_apr, trainLabels_apr)

    fig, ax_det = plt.subplots()
    plot_det_curve(model_acc, testData_acc, testLabels_acc, ax=ax_det, name="Accurate")
    plot_det_curve(model_apr, testData_apr, testLabels_apr, ax=ax_det, name="Approximate")

    ax_det.set_title('Detection Error Tradeoff (DET) curves')
    ax_det.grid(linestyle='--')

    plt.legend()
    plt.show()

#plotCurves()
trainAndSave(positiveFiles_acc,negativeFiles_acc,"acc_classifier_graz_L1")
trainAndSave(positiveFiles_apr,negativeFiles_apr,"apr_classifier_graz_L1")


