
import csv
import numpy as np

def readFeatureFromTxt(data_path):
    featureDict = dict()
    with open(data_path, "r") as featurefile:
        n = 0;
        for line in featurefile:
            if n == 0:
                n = n + 1
                continue
            else:
                tabs = line.split("\t")
                medcode = int(tabs[0])
                features = [int(a) for a in tabs[2:]]
                featureDict[medcode] = features
                n = n + 1
    return featureDict


def readFeatureFromCSV(data_path):
    featureDict = dict()
    with open(data_path, "r") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
        n = 0;
        for line in spamreader:
            if n == 0:
                featurecodes = line[1:]
                n = n + 1
            else:
                medcode = int(line[0])
                sims = [float(a) for a in line[1:]]

                featureDict[medcode] = sims
                n = n + 1
        print(data_path)
        print(len(sims))
    return featurecodes, featureDict

# def writeFeatureFile(datapath, outputpath, nonEmptyList):
#     featureCodes, circDict = readFeatureFromCSV(datapath)
#     sortedCirc = sorted(circDict.items(), key=lambda x: x[0])
#     circFeatureList = []
#     circCodeList = []
#     for code, feature in sortedCirc:
#         circFeatureList.append(feature)
#         circCodeList.append(code)
#     strCodeList = [str(code) for code in circCodeList]
#
#     with open(outputpath, "wb") as csvfile:
#         csvfile.write("," + ",".join(featureCodes) + "\n")
#         for i in nonEmptyList:
#             strSim = [str(circFeatureList[i][j]) for j in nonEmptyList]
#             csvfile.write(strCodeList[i] + "," + ",".join(strSim) + "\n")


def writeSimFile(datapath, outputpath, nonEmptyList=None):
    _, circDict = readFeatureFromCSV(datapath)
    sortedCirc = sorted(circDict.items(), key=lambda x: x[0])
    circFeatureList = []
    circCodeList = []
    for code, feature in sortedCirc:
        circFeatureList.append(feature)
        circCodeList.append(code)
    strCodeList = [str(code) for code in circCodeList]
    circSIM = np.zeros((len(circCodeList), len(circCodeList)))
    for i in range(len(circFeatureList)):
        for j in range(i):
            coNum = len([t for t in range(len(circFeatureList[i])) if
                         circFeatureList[i][t] == 1 and circFeatureList[j][t] == 1])
            circSIM[i, j] = float(coNum) / float(sum(circFeatureList[i]) + sum(circFeatureList[j]) - coNum)
            circSIM[j, i] = circSIM[i, j]
    if nonEmptyList is None:
        with open(outputpath, "wb") as csvfile:
            csvfile.write("," + ",".join(strCodeList) + "\n")
            for i in range(len(circCodeList)):
                strSim = [str(sim) for sim in circSIM[i, :]]
                csvfile.write(strCodeList[i] + "," + ",".join(strSim) + "\n")
    else:
        with open(outputpath, "wb") as csvfile:
            csvfile.write("," + ",".join([strCodeList[j] for j in nonEmptyList]) + "\n")
            for i in nonEmptyList:
                strSim = [str(sim) for sim in circSIM[i, nonEmptyList]]
                csvfile.write(strCodeList[i] + "," + ",".join(strSim) + "\n")

def cosineSim(list1, list2):
    a = np.array(list1)
    b = np.array(list2)
    ab = np.dot(a,b)
    a2 = np.linalg.norm(a)
    b2 = np.linalg.norm(b)
    return ab/(a2*b2)

def rbf(list1, list2):
    a = np.array(list1)
    b = np.array(list2)
    n = len(a)
    c = np.linalg.norm(a-b)
    d = np.exp(-c*c/n)
    if d <0.00005:
        d =0
    return d

def cpi(datapath, outputpath, nonEmptyList=None):
    _, circDict = readFeatureFromCSV(datapath)
    sortedCirc = sorted(circDict.items(), key=lambda x: x[0])
    circFeatureList = []
    circCodeList = []
    for code, feature in sortedCirc:
        circFeatureList.append(feature)
        circCodeList.append(code)
    print("cpi")
    print(len(circFeatureList[0]))
    strCodeList = [str(code) for code in circCodeList]
    circSIM = np.zeros((len(circCodeList), len(circCodeList)))
    for i in range(len(circFeatureList)):
        for j in range(i):
            #circSIM[i, j] = cosineSim(circFeatureList[i], circFeatureList[j])
            circSIM[i, j] = rbf(circFeatureList[i], circFeatureList[j])
            circSIM[j, i] = circSIM[i, j]
    if nonEmptyList is None:
        with open(outputpath, "wb") as csvfile:
            csvfile.write("," + ",".join(strCodeList) + "\n")
            for i in range(len(circCodeList)):
                strSim = [str(sim) for sim in circSIM[i, :]]
                csvfile.write(strCodeList[i] + "," + ",".join(strSim) + "\n")
    else:
        with open(outputpath, "wb") as csvfile:
            csvfile.write("," + ",".join([strCodeList[j] for j in nonEmptyList]) + "\n")
            for i in nonEmptyList:
                strSim = [str(sim) for sim in circSIM[i, nonEmptyList]]
                csvfile.write(strCodeList[i] + "," + ",".join(strSim) + "\n")


def readLabel(datapath, featurepath, outputpath):
    ddiset = set()
    with open(datapath,"rb") as f:
        for line in f:
            tabs = line.strip().split("\t")
            ddiset.add((tabs[0], tabs[1]))
            ddiset.add((tabs[1], tabs[0]))
    _, circDict = readFeatureFromCSV(featurepath)
    sortedCodes = sorted(circDict.keys())#int
    listCodes = [str(code) for code in sortedCodes]#string
    ddiMAT = np.zeros((len(listCodes), len(listCodes)))
    for i in range(len(listCodes)):
        for j in range(len(listCodes)):
            if (listCodes[i],listCodes[j]) in ddiset:
                ddiMAT[i,j] = 1
    with open(outputpath, "wb") as csvfile:
        csvfile.write("," + ",".join(listCodes) + "\n")
        NonEmptySet=[]
        for i in range(len(listCodes)):
            if sum(ddiMAT[i,:])==0:
                continue
            else:
                NonEmptySet.append(i)
        for i in NonEmptySet:
            strSim = [str(sim) for sim in ddiMAT[i, NonEmptySet]]
            csvfile.write(listCodes[i] + "," + ",".join(strSim) + "\n")
    return NonEmptySet

def readLabel2(datapath, featurepath, outputDir, label):
    labelDDIset = set()
    ddiset = set()
    with open(datapath, "rb") as f:
        for line in f:
            tabs = line.strip().split("\t")
            ddiset.add((tabs[0], tabs[1]))
            ddiset.add((tabs[1], tabs[0]))
            if tabs[2]==label:
                labelDDIset.add((tabs[0], tabs[1]))
                labelDDIset.add((tabs[1], tabs[0]))
    _, circDict = readFeatureFromCSV(featurepath)
    sortedCodes = sorted(circDict.keys())  # int
    listCodes = [str(code) for code in sortedCodes]  # string
    ddiMAT = np.zeros((len(listCodes), len(listCodes)))
    labelDdiMAT = np.zeros((len(listCodes), len(listCodes)))
    for i in range(len(listCodes)):
        for j in range(len(listCodes)):
            if (listCodes[i], listCodes[j]) in ddiset:
                ddiMAT[i, j] = 1
            if (listCodes[i], listCodes[j]) in labelDDIset:
                labelDdiMAT[i,j]=1
    with open(outputDir+"ddi_"+label+".csv", "wb") as csvfile:
        csvfile.write("," + ",".join(listCodes) + "\n")
        NonEmptySet = []
        for i in range(len(listCodes)):
            if sum(ddiMAT[i, :]) == 0:
                continue
            else:
                NonEmptySet.append(i)
        print("Number of Codes:", len(NonEmptySet))
        for i in NonEmptySet:
            strSim = [str(sim) for sim in labelDdiMAT[i, NonEmptySet]]
            csvfile.write(listCodes[i] + "," + ",".join(strSim) + "\n")
    return NonEmptySet

def checkNUMadr():
    listCodes = set()
    with open("../data/CIDs","r") as f:
        for line in f:
            if len(line.strip())>0:
                listCodes.add(line.strip())
    print(len(listCodes))
    listADR = set()
    with open("../data/ddi.txt","r") as f:
        for line in f:
            tabs = line.strip().split("\t")
            if tabs[0] in listCodes and tabs[1] in listCodes:
                listADR.add(tabs[2])
    print(len(listADR))

def filterFeatures(data_path, outputpath):
    featureDict = set()
    with open("../data/CIDs") as f:
        for line in f:
            if line.strip()>0:
                featureDict.add(line.strip())
    wf  = open(outputpath, "wb")
    with open(data_path, "r") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
        n = 0;
        for line in spamreader:
            if n == 0:
                featurecodes = line[1:]
                n = n + 1
                wf.write(",".join(line)+"\n")
            else:
                medcode = line[0]
                if medcode in featureDict:
                    wf.write(",".join(line)+"\n")
                n = n + 1
    wf.close()
    return featureDict

if __name__ == '__main__':
    featurepath = "../data/cpi.csv"
    labelpath = "../data/ddi.txt"
    filterFeatures(featurepath, "./Dataset/cpi.csv")

    # nonEmptyList = readLabel(labelpath, featurepath, "./Dataset/ddi.csv")
    #
    # data_path = "../data/cpi.csv"
    # cpi(data_path,  "./Dataset/cpi_sim.csv", nonEmptyList)
    #
    # data_path = "../data/indication.csv"
    # outputpath = './Dataset/indication_sim.csv'
    # writeSimFile(data_path, outputpath, nonEmptyList)
    # # readFeatureFromCSV(data_path)
    #
    # data_path = "../data/pubchem.csv"
    # outputpath = './Dataset/pubchem_sim.csv'
    # writeSimFile(data_path, outputpath, nonEmptyList)
    # # readFeatureFromCSV(data_path)
    #
    # data_path = "../data/TTDS.csv"
    # outputpath = './Dataset/TTDS_sim.csv'
    # writeSimFile(data_path, outputpath, nonEmptyList)
    # # readFeatureFromCSV(data_path)
    #
    # labelName = "C0917801"
    # readLabel2(labelpath, featurepath, "./Dataset/",labelName)

    #checkNUMadr()




