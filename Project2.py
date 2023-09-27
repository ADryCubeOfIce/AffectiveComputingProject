from sklearn import svm, tree
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
import os
import numpy as np
import sys
import statistics
import matplotlib.pyplot as plt
import random

def BoxPlots(data, outFile):
    fig = plt.figure(figsize =(10, 7))
 
    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    
    meanData = []
    varData = []
    minData = []
    maxData = []
    for point in data:
        meanData.append(point[0])
        varData.append(point[1])
        minData.append(point[2])
        maxData.append(point[3])
    
    allData = [meanData, varData, minData, maxData]
    # Creating plot
    bp = ax.boxplot(allData)
    
    # show plot
    plt.savefig(outFile)
    return

def PrintEvalMetrics(pred, indices, y):
    #manually merge predictions and testing labels from each of the folds to make confusion matrix
    finalPredictions = []
    groundTruth = []
    for p in pred:
        finalPredictions.extend(p)
    for i in indices:
        groundTruth.extend(y[i])
    print(confusion_matrix(finalPredictions, groundTruth))
    print("Precision: ", precision_score(groundTruth, finalPredictions, average='macro'))
    print("Recall: ", recall_score(groundTruth, finalPredictions, average='macro'))
    print("Accuracy: " , accuracy_score(groundTruth, finalPredictions))

def ReadData(path, type):
    subjects = []
    dataList = []
    labels = []
    index = 1
    subjectAll = []
    with open(path) as F:
        for l in F.readlines():
            line = l.replace("\n", "")
            line = line.split(",")
            information = line[:3]
            data = line[3:]
            information[2] = information[2].replace("No Pain", "0")
            information[2] = information[2].replace("Pain", "1")
            information[1] = information[1].replace("BP Dia_mmHg", "dia")
            information[1] = information[1].replace("EDA_microsiemens", "eda")
            information[1] = information[1].replace("LA Systolic BP_mmHg", "sys")
            information[1] = information[1].replace("Respiration Rate_BPM", "res")
            
            numData = [float(point) for point in data]
            maxData = max(numData)
            minData = min(numData)
            meanData = statistics.mean(numData)
            varianceData = statistics.variance(numData, meanData)
            
            stats = [meanData, varianceData, minData, maxData]
            
            if(information[1] == "dia" and type == "dia"):
                dataList.append(stats)
                labels.append(int(information[2]))
            elif(information[1] == "eda" and type == "eda"):
                dataList.append(stats)
                labels.append(int(information[2]))
            elif(information[1] == "sys" and type == "sys"):
                dataList.append(stats)
                labels.append(int(information[2]))
            elif(information[1] == "res" and type == "res"):
                dataList.append(stats)
                labels.append(int(information[2]))
            elif(type == "all"):
                for stat in stats:
                    subjectAll.append(stat)
                if(information[1] == "res"):
                    dataList.append(subjectAll)
                    labels.append(int(information[2]))
                    subjectAll = []
            if(index % 8 == 1):
                subjectID = information[0]
                subjects.append(subjectID)
            index += 1
    return subjects, dataList, labels

def CrossFoldValidation(data, labels, subjects, classifier = "SVM"):
    # change lists to np arrays
    X = np.array(data)
    y = np.array(labels)
    clf = None
    if classifier == "SVM":
        # default SVM 
        clf = svm.SVC()
    elif classifier == "RF":
        # default random forest
        clf = RandomForestClassifier()
    elif classifier == "TREE":
        # default random forest
        clf = tree.DecisionTreeClassifier()
    # save predictions and indices
    pred=[]
    test_indices=[]

    # index the list in X
    Z = np.array(subjects)
    indices = []
    for z in range(len(Z)):
        indices.append(z)
        
    random.shuffle(indices)

    # split X into 10 groups
    groups = np.array_split(indices, 10)
    # convert the groups back into lists
    groups = [fold.tolist() for fold in [*groups]]

    # for each group
    for i in range(len(groups)):
        # create the set of 9 training sets and 1 test set
        trainSet1 = groups[:i]
        trainSet2 = groups[i + 1:]
        trainSet = trainSet1 + trainSet2
        testSet = groups[i:i + 1]
    
        training = []
        testing = []
        # make train set a single set
        for element in trainSet:
            for i in element:
                training.append(i*2)
                training.append(i*2 + 1)

        # make sure test set is formatted properly
        for element in testSet:
            for i in element:
                testing.append(i*2)
                testing.append(i*2 + 1)

        # convert back to np array
        train_index = np.array(training)
        test_index = np.array(testing)

        #train classifier
        clf.fit(X[train_index], y[train_index])
        #get predictions and save
        pred.append(clf.predict(X[test_index]))
        #save current test index
        test_indices.append(test_index)
    return pred, test_indices, y

# set each command line argument
dataType = sys.argv[1]
dataPath = sys.argv[2] # './Project2Data.csv'
dataClass = "RF"  # sys.argv[3], RF was determined to be the best classifier

subjects, data, labels = ReadData(dataPath, dataType)
pred, test_indicies, y = CrossFoldValidation(data, labels, subjects, dataClass)
PrintEvalMetrics(pred, test_indicies, y)

# dataType = "dia"
# dataPath = './Project2Data.csv'
# dataClass = "SVM"
# print("SVM - dia")
# subjects, data, labels = ReadData(dataPath, dataType)
# pred, test_indicies, y = CrossFoldValidation(data, labels, subjects, dataClass)
# PrintEvalMetrics(pred, test_indicies, y)

# dataType = "eda"
# print("SVM - eda")
# subjects, data, labels = ReadData(dataPath, dataType)
# pred, test_indicies, y = CrossFoldValidation(data, labels, subjects, dataClass)
# PrintEvalMetrics(pred, test_indicies, y)

# dataType = "sys"
# print("SVM - sys")
# subjects, data, labels = ReadData(dataPath, dataType)
# pred, test_indicies, y = CrossFoldValidation(data, labels, subjects, dataClass)
# PrintEvalMetrics(pred, test_indicies, y)

# dataType = "res"
# print("SVM - res")
# subjects, data, labels = ReadData(dataPath, dataType)
# pred, test_indicies, y = CrossFoldValidation(data, labels, subjects, dataClass)
# PrintEvalMetrics(pred, test_indicies, y)

# dataType = "all"
# print("SVM - all")
# subjects, data, labels = ReadData(dataPath, dataType)
# pred, test_indicies, y = CrossFoldValidation(data, labels, subjects, dataClass)
# PrintEvalMetrics(pred, test_indicies, y)

# dataType = "dia"
# dataClass = "RF"
# print("RF - dia")
# subjects, data, labels = ReadData(dataPath, dataType)
# pred, test_indicies, y = CrossFoldValidation(data, labels, subjects, dataClass)
# PrintEvalMetrics(pred, test_indicies, y)
# BoxPlots(data, 'diaBoxPlot.png')

# dataType = "eda"
# print("RF - eda")
# subjects, data, labels = ReadData(dataPath, dataType)
# pred, test_indicies, y = CrossFoldValidation(data, labels, subjects, dataClass)
# PrintEvalMetrics(pred, test_indicies, y)
# BoxPlots(data, 'edaBoxPlot.png')

# dataType = "sys"
# print("RF - sys")
# subjects, data, labels = ReadData(dataPath, dataType)
# pred, test_indicies, y = CrossFoldValidation(data, labels, subjects, dataClass)
# PrintEvalMetrics(pred, test_indicies, y)
# BoxPlots(data, 'sysBoxPlot.png')

# dataType = "res"
# print("RF - res")
# subjects, data, labels = ReadData(dataPath, dataType)
# pred, test_indicies, y = CrossFoldValidation(data, labels, subjects, dataClass)
# PrintEvalMetrics(pred, test_indicies, y)
# BoxPlots(data, 'resBoxPlot.png')

# dataType = "all"
# print("RF - all")
# subjects, data, labels = ReadData(dataPath, dataType)
# pred, test_indicies, y = CrossFoldValidation(data, labels, subjects, dataClass)
# PrintEvalMetrics(pred, test_indicies, y)

# dataType = "dia"
# dataClass = "TREE"
# print("TREE - dia")
# subjects, data, labels = ReadData(dataPath, dataType)
# pred, test_indicies, y = CrossFoldValidation(data, labels, subjects, dataClass)
# PrintEvalMetrics(pred, test_indicies, y)

# dataType = "eda"
# print("TREE - eda")
# subjects, data, labels = ReadData(dataPath, dataType)
# pred, test_indicies, y = CrossFoldValidation(data, labels, subjects, dataClass)
# PrintEvalMetrics(pred, test_indicies, y)

# dataType = "sys"
# print("TREE - sys")
# subjects, data, labels = ReadData(dataPath, dataType)
# pred, test_indicies, y = CrossFoldValidation(data, labels, subjects, dataClass)
# PrintEvalMetrics(pred, test_indicies, y)

# dataType = "res"
# print("TREE - res")
# subjects, data, labels = ReadData(dataPath, dataType)
# pred, test_indicies, y = CrossFoldValidation(data, labels, subjects, dataClass)
# PrintEvalMetrics(pred, test_indicies, y)

# dataType = "all"
# print("TREE - all")
# subjects, data, labels = ReadData(dataPath, dataType)
# pred, test_indicies, y = CrossFoldValidation(data, labels, subjects, dataClass)
# PrintEvalMetrics(pred, test_indicies, y)

