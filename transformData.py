import numpy as np
import pickle as pkl
import csv
import random

def readFromCSV(data_path):
    adj = []
    with open(data_path,"r") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
        n =0;
        for line in spamreader:
            if n==0:
                codes = [int(str(a).lstrip("CID")) for a in line[1:]]
                n = n+1
            else:
                code = line[0]
                sims = [float(a) for a in line[1:]]
                adj.append(sims)
                n = n +1
    return codes, np.array(adj)

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

    return featurecodes, featureDict


def readFeature(data_path):
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

def list2dict(codes):
    dic = dict()
    for i in range(len(codes)):
        dic[codes[i]] = i
    return dic

def tranformFormat(codes, adj, featureDict= None):
    codeDic = list2dict(codes)
    numCode = len(codes)
    graph = dict()
    if featureDict is None:
        x = np.zeros((numCode, numCode))
        for code in codes:
            index = codeDic[code]
            x[index,index] =1.0
            graph[index] = [gi for gi in range(len(codes)) if adj[index, gi]>0 and gi!=index]
        return x, graph
    else:
        x = list()
        n = 0
        for code in codes:
            feature = featureDict[code]
            x.append(feature)
            index = len(x) - 1
            graph[index] = [gi for gi in range(len(codes)) if adj[index, gi] > 0 and gi != index]


        return np.array(x), graph



def splitData_node(y, test_ratio):
    #test_ratio: ratio on all data including val
    numNode = len(y)
    Indexs = random.sample(range(numNode), int((test_ratio + 0.1) * numNode))
    valIndex = random.sample(Indexs, int(0.1 * numNode))
    testIndex = [i for i in Indexs if i not in valIndex]
    trainIndex = [i for i in range(numNode) if i not in Indexs]
    trainMask = np.zeros((numNode, numNode))

    # test_ratio: ratio on data other than val.
    # numNode = len(y)
    # valIndex = random.sample(range(numNode), int(0.1 * numNode))
    # Indexs = [i for i in range(numNode) if i not in valIndex]
    # numIndex = len(Indexs)
    #
    # testIndex = random.sample(Indexs,int((test_ratio) * numIndex))
    # trainIndex = [i for i in Indexs if i not in testIndex]
    # trainMask = np.zeros((numNode,numNode))
    for i in trainIndex:
        for j in trainIndex:
            if j<i:
                continue
            elif y[i,j]==1:
                trainMask[i,j] = 1
    for i in trainIndex:
        for j in trainIndex:
            if j<i:
                continue
            elif y[i,j]==0:
                sig = random.random()
                if sig>0.6:
                    trainMask[i, j] = 1


    trainMask = trainMask+trainMask.T

    valMask = np.zeros((numNode, numNode))
    for i in valIndex:
        for j in valIndex:
            if j < i:
                continue
            else:
                valMask[i, j] = 1
                valMask[j,i] = 1
    for i in valIndex:
        for j in trainIndex:
            valMask[i, j] = 1
            valMask[j,i] = 1

    testMask = np.zeros((numNode,numNode))
    for i in testIndex:
        for j in testIndex:
            if j<i:
                continue
            else:
                testMask[i,j] = 1
                testMask[j,i] = 1
    for i in testIndex:
        for j in trainIndex:
            testMask[i, j] = 1
            testMask[j, i] = 1

    return trainMask, valMask, testMask

def splitData_edge(y, test_ratio):
    numNode = len(y)
    population = [(i,j) for i in range(numNode) for j in range(numNode) if i>j]
    numPopulation = len(population)
    choices = random.sample(population, int(numPopulation*(test_ratio+0.1)))
    valchoices = random.sample(choices, int(numPopulation*0.1))
    testMask = np.zeros((numNode,numNode))
    valMask = np.zeros((numNode, numNode))
    for (i,j) in choices:
        if (i,j) not in valchoices:
            testMask[i,j] = 1
            testMask[j,i] = 1
        else:
            valMask[i,j] = 1
            valMask[j, i] = 1
    trainMask = np.ones((numNode,numNode)) - testMask - valMask
    np.fill_diagonal(trainMask,0)

    # train_ratio = 0.9 - test_ratio
    # numNode = len(y)
    # population = [(i, j) for i in range(numNode) for j in range(numNode) if i > j and y[i,j]==1]
    # numPopulation = len(population)
    # numNode = len(y)
    # choices = random.sample(population, int(numPopulation * (train_ratio + 0.1)))
    # valchoices = random.sample(choices, int(numPopulation * 0.1))
    # testchoices = [(i,j) for (i,j) in population if (i,j) not in choices]
    # trainMask = np.zeros((numNode,numNode))
    # valMask = np.zeros((numNode, numNode))
    # testMask = np.zeros((numNode, numNode))
    # for (i,j) in choices:
    #     if (i,j) not in valchoices:
    #         trainMask[i,j] = 1
    #         trainMask[j,i] = 1
    #     else:
    #         valMask[i,j] = 1
    #         valMask[j,i] = 1
    # for (i,j) in testchoices:
    #     testMask[i,j] = 1
    #     testMask[j,i] = 1
    #
    # nolink_population = [(i, j) for i in range(numNode) for j in range(numNode) if i > j and y[i, j] == 0]
    # numNolink = len(nolink_population)
    # choices = random.sample(nolink_population, int(numPopulation * (train_ratio + 0.1)))
    # valchoices = random.sample(choices, int(numPopulation * 0.1))
    # testchoices = [(i, j) for (i, j) in nolink_population if (i, j) not in choices]
    # testchoices = random.sample(testchoices, int(numPopulation * test_ratio))
    # for (i, j) in choices:
    #     if (i, j) not in valchoices:
    #         trainMask[i, j] = 1
    #         trainMask[j, i] = 1
    #     else:
    #         valMask[i, j] = 1
    #         valMask[j, i] = 1
    # for (i, j) in testchoices:
    #     testMask[i, j] = 1
    #     testMask[j, i] = 1

    return trainMask, testMask, valMask

def writeSimFile(datapath):
    circDict = readFeature(datapath)
    sortedCirc = sorted(circDict.items(), key=lambda x: x[0])
    circFeatureList = []
    circCodeList = []
    for code, feature in sortedCirc:
        circFeatureList.append(feature)
        circCodeList.append(code)
    strCodeList = ["CID" + ("0000000" + str(code))[-9:] for code in circCodeList]
    circSIM = np.zeros((len(circCodeList), len(circCodeList)))
    for i in range(len(circFeatureList)):
        for j in range(i):
            coNum = len([t for t in range(len(circFeatureList[i])) if
                         circFeatureList[i][t] == 1 and circFeatureList[j][t] == 1])
            circSIM[i, j] = float(coNum) / float(sum(circFeatureList[i]) + sum(circFeatureList[j]) - coNum)
            circSIM[j, i] = circSIM[i, j]
    with open("../Dataset/circular_sim.csv", "wb") as csvfile:
        csvfile.write("," + ",".join(strCodeList) + "\n")
        for i in range(len(circCodeList)):
            strSim = [str(sim) for sim in circSIM[i, :]]
            csvfile.write(strCodeList[i] + "," + ",".join(strSim) + "\n")


def run_569():
    data_path1 = "../Dataset/chem_sim.csv"
    data_path2 = "../Dataset/circular_sim.csv"
    data_path3 = "../Dataset/label_sim.csv"
    data_path4 = "../Dataset/offlabel_sim.csv"
    label_path = "../Dataset/ddi_569.csv"
    feature_path = "../Dataset/newsample-pubchem.txt"
    circular_path = "../Dataset/newsample-circular.txt"
    codes, adj_1 = readFromCSV(data_path1)
    _, adj_2 = readFromCSV(data_path2)
    _, adj_3 = readFromCSV(data_path3)
    _, adj_4 = readFromCSV(data_path4)

    printCIDs = [str(a) for a in codes]
    wf = open("../Dataset/cid.txt","wb")
    wf.write(", ".join(printCIDs))
    wf.close()


    featureDict = readFeature(circular_path)
    x_featureless, graph_1 = tranformFormat(codes, adj_1)
    _, graph_2 = tranformFormat(codes, adj_2)
    _, graph_3 = tranformFormat(codes, adj_3, featureDict)
    x, graph_4 = tranformFormat(codes, adj_4)


    codes, y = readFromCSV(label_path)
    #trainMask, testMask, valMask = splitData_edge(y, 0.5)
    trainMask, testMask, valMask = splitData_node(y, 0.75)

    pkl.dump(x,open("DDIdata/ind.ddi.allx","wb"))
    pkl.dump(y, open("DDIdata/ind.ddi.ally","wb"))
    pkl.dump(x_featureless, open("DDIdata/ind.ddi.x_featureless", "wb"))

    pkl.dump(trainMask, open("DDIdata/ind.ddi.trainMask", "wb"))
    pkl.dump(testMask, open("DDIdata/ind.ddi.testMask", "wb"))
    pkl.dump(valMask, open("DDIdata/ind.ddi.valMask", "wb"))
    pkl.dump([adj_1, adj_2, adj_3, adj_4], open("DDIdata/ind.ddi.adjmat","wb"))
    pkl.dump([graph_1, graph_2, graph_3, graph_4], open("DDIdata/ind.ddi.graph","wb"))

    # print(len(codes))

def run_526():
    data_path1 = "../Dataset/pubchem_sim.csv"
    data_path2 = "../Dataset/indication_sim.csv"
    data_path3 = "../Dataset/TTDS_sim.csv"
    data_path4 = "../Dataset/cpi_sim.csv"
    label_path = "../Dataset/ddi_C0917801.csv"
    feature_path = "../Dataset/pubchem.csv"
    codes, adj_1 = readFromCSV(data_path1)
    _, adj_2 = readFromCSV(data_path2)
    _, adj_3 = readFromCSV(data_path3)
    _, adj_4 = readFromCSV(data_path4)

    # printCIDs = [str(a) for a in codes]
    # wf = open("../Dataset/cid.txt","wb")
    # wf.write(", ".join(printCIDs))
    # wf.close()


    _,featureDict = readFeatureFromCSV(feature_path)
    x_featureless, graph_1 = tranformFormat(codes, adj_1)
    _, graph_2 = tranformFormat(codes, adj_2)
    x, graph_3 = tranformFormat(codes, adj_3, featureDict)
    _, graph_4 = tranformFormat(codes, adj_4)


    codes, y = readFromCSV(label_path)
    #trainMask, testMask, valMask = splitData_edge(y, 0.5)
    trainMask, testMask, valMask = splitData_node(y, 0.25)

    pkl.dump(x,open("DDIdata/ind.ddi.allx","wb"))
    pkl.dump(y, open("DDIdata/ind.ddi.ally","wb"))
    pkl.dump(x_featureless, open("DDIdata/ind.ddi.x_featureless", "wb"))

    pkl.dump(trainMask, open("DDIdata/ind.ddi.trainMask", "wb"))
    pkl.dump(testMask, open("DDIdata/ind.ddi.testMask", "wb"))
    pkl.dump(valMask, open("DDIdata/ind.ddi.valMask", "wb"))
    pkl.dump([adj_1, adj_2, adj_3, adj_4], open("DDIdata/ind.ddi.adjmat","wb"))
    pkl.dump([graph_1, graph_2, graph_3, graph_4], open("DDIdata/ind.ddi.graph","wb"))

    # print(len(codes))


def run_for_MLK():
    data_path1 = "../Dataset/pubchem_sim.csv"
    data_path2 = "../Dataset/indication_sim.csv"
    data_path3 = "../Dataset/TTDS_sim.csv"
    data_path4 = "../Dataset/cpi_sim.csv"
    label_path = "../Dataset/ddi.csv"
    feature_path1 = "../Dataset/pubchem.csv"
    feature_path2 = "../Dataset/indication.csv"
    feature_path3 = "../Dataset/TTDS.csv"
    feature_path4 = "../Dataset/cpi.csv"
    codes, adj_1 = readFromCSV(data_path1)
    _, adj_2 = readFromCSV(data_path2)
    _, adj_3 = readFromCSV(data_path3)
    _, adj_4 = readFromCSV(data_path4)

    # printCIDs = [str(a) for a in codes]
    # wf = open("../Dataset/cid.txt","wb")
    # wf.write(", ".join(printCIDs))
    # wf.close()


    _, featureDict1 = readFeatureFromCSV(feature_path1)
    _, featureDict2 = readFeatureFromCSV(feature_path2)
    _, featureDict3 = readFeatureFromCSV(feature_path3)
    _, featureDict4 = readFeatureFromCSV(feature_path4)
    featureDict = dict()
    for code in featureDict1.keys():
        x1 = featureDict1[code]
        x2 = featureDict2[code]
        x3 = featureDict3[code]
        x4 = featureDict4[code]
        featureDict[code] = x1+x2+x3+x4

    x_featureless, graph_1 = tranformFormat(codes, adj_1)
    _, graph_2 = tranformFormat(codes, adj_2)
    x, graph_3 = tranformFormat(codes, adj_3, featureDict)
    _, graph_4 = tranformFormat(codes, adj_4)

    codes, y = readFromCSV(label_path)
    # trainMask, testMask, valMask = splitData_edge(y, 0.5)
    trainMask, testMask, valMask = splitData_node(y, 0.5)

    pkl.dump(x, open("DDIdata/ind.ddi.allx", "wb"))
    pkl.dump(y, open("DDIdata/ind.ddi.ally", "wb"))
    pkl.dump(x_featureless, open("DDIdata/ind.ddi.x_featureless", "wb"))

    pkl.dump(trainMask, open("DDIdata/ind.ddi.trainMask", "wb"))
    pkl.dump(testMask, open("DDIdata/ind.ddi.testMask", "wb"))
    pkl.dump(valMask, open("DDIdata/ind.ddi.valMask", "wb"))
    pkl.dump([adj_1, adj_2, adj_3, adj_4], open("DDIdata/ind.ddi.adjmat", "wb"))
    pkl.dump([graph_1, graph_2, graph_3, graph_4], open("DDIdata/ind.ddi.graph", "wb"))

    # print(len(codes))

if __name__ == '__main__':
    run_569()
    # run_526()
    # run_for_MLK()