from transformData import *
import pickle as pkl
import sys
import heapq


def ddi_load_data_GAE(dataset_str):
    """Load data."""
    names = ['allx', 'ally', 'graph',"adjmat", "trainMask", "valMask", "testMask"]
    objects = []
    for i in range(len(names)):
        with open("DDIdata/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, graphs, adjmats, train_mask, val_mask, test_mask = tuple(objects)

    tmpx = np.multiply(y,train_mask+train_mask.T)

    #features = sp.coo_matrix(tmpx).tolil()
    features = tmpx

    adjs = []
    for adjmat in adjmats:
        adjs.append(adjmat)

    return adjs, features, tmpx, y, train_mask, val_mask, test_mask


def simCodes():
    # Load data
    adjs, densefeatures, x, y, train_mask, val_mask, test_mask = ddi_load_data_GAE("ddi")
    print(len(x))

    resultPath = "resOUTPUT.pkl"
    testsubs, preds, labels, mixedADJ = pkl.load(open(resultPath, "rb"))
    testSubDic = [(testsubs[0][i], testsubs[1][i]) for i in range(len(testsubs[0]))]

    data_path1 = "../Dataset/chem_sim.csv"
    codes, adj_1 = readFromCSV(data_path1)
    subs = np.where(adj_1 > 0.5)
    adj_2 = adjs[1]

    for i in range(len(subs[0])):
        if subs[0][i] < subs[1][i] and (subs[0][i], subs[1][i]) not in testSubDic:
            # print(subs[0][i], subs[1][i])
            # print(preds[subs[0][i]])
            # print(preds[subs[1][i]])
            selvar = [adjs[j][subs[0][i], subs[1][i]] for j in range(len(adjs))]
            if sum(selvar) < 2.3:
                continue
            print("CODE pairs, chem_sim, circular_sim, label_sim, offlabel_sim, difference of ddi")
            print(codes[subs[0][i]], codes[subs[1][i]])
            for j in range(len(adjs)):
                print(adjs[j][subs[0][i], subs[1][i]])
            diff = y[subs[0][i]] - y[subs[1][i]]
            print("Difference bwtween the two codes:", len(np.nonzero(diff)[0]))
            print

    print("Ready!")

def compareTWO(code1, code2):#for some specific drugs
    # Load data
    adjs, densefeatures, x, y, train_mask, val_mask, test_mask = ddi_load_data_GAE("ddi")
    print(code1, code2)

    resultPath = "resOUTPUT.pkl"
    testsubs, preds, labels, mixedADJ, attentions = pkl.load(open(resultPath, "rb"))

    mixedADJ_LP = pkl.load(open("mixedADJ_LP.pkl", "rb"))
    # mixedADJ_GCN = pkl.load(open("mixedADJ_GCN.pkl", "rb"))
    attention_Semi = pkl.load(open("attention_Semi.pkl", "rb"))

    data_path1 = "../Dataset/222backup/pubchem_sim.csv"
    codes, adj_1 = readFromCSV(data_path1)

    ind1 = codes.index(code1)
    ind2 = codes.index(code2)
    sum =0.0
    ADJ = np.zeros_like(adjs[0])
    ADJs = []
    for i in range(len(adjs)):
        print(adjs[i][ind1,ind2])
        print(attentions[i][ind1])
        sum = sum + adjs[i][ind1,ind2] *attentions[i][ind1]
        ADJ = ADJ + np.matmul(np.diag(attentions[i]), adjs[i])
        ADJs.append(np.matmul(np.diag(attentions[i]), adjs[i]))
    print("mixed:",(mixedADJ[ind1,ind2] + mixedADJ[ind2,ind1])/2.0)
    # print(sum)
    print(ADJ[ind1,ind2])


    # print("GCN:", mixedADJ_GCN[ind1, ind2])
    # print("LP",mixedADJ_LP[ind1,ind2])
    # print("semiGAE:", (mixedADJ_Semi[ind1, ind2] + mixedADJ_Semi[ind2, ind1]) / 2.0)

    # diff = y[ind1]-y[ind2]
    # print("Different DDIs bwtween the two codes:", len(np.nonzero(diff)[0]))

    print

if __name__ == '__main__':
    # # Load data
    # adjs, densefeatures, x, y, train_mask, val_mask, test_mask = ddi_load_data_GAE("ddi")
    # print(len(x))
    #
    # resultPath = "resOUTPUT.pkl"
    # testsubs, preds, labels, mixedADJ = pkl.load(open(resultPath, "rb"))
    #
    # data_path1 = "../Dataset/222backup/pubchem_sim.csv"
    # codes, adj_1 = readFromCSV(data_path1)
    #
    # testlabels = labels[testsubs]
    # testpreds = preds[testsubs]
    # indexedPreds = list(enumerate(testpreds))
    # topk = heapq.nlargest(50, indexedPreds, key=lambda d:d[1])
    # for k, v in topk:
    #     position = (testsubs[0][k], testsubs[1][k])
    #     if testlabels[k] ==0:
    #         print(codes[testsubs[0][k]], codes[testsubs[1][k]])
    # print("Ready!")


    testCodes = [(2022, 3454), (2118, 3261), (2118, 5556), (2160,2895),(2369,4171)]
    for (code1, code2) in testCodes:
        compareTWO(code1,code2)


    # # Load data
    # adjs, densefeatures, x, y, train_mask, val_mask, test_mask = ddi_load_data_GAE("ddi")
    # print(len(x))
    #
    # resultPath = "resOUTPUT.pkl"
    # testsubs, preds, labels, mixedADJ = pkl.load(open(resultPath, "rb"))
    #
    # data_path1 = "../Dataset/chem_sim.csv"
    # codes, adj_1 = readFromCSV(data_path1)
    #
    # testlabels = labels[testsubs]
    # testpreds = preds[testsubs]
    # indexedPreds = list(enumerate(testpreds))
    # topk = heapq.nlargest(50, indexedPreds, key=lambda d:d[1])
    # for k, v in topk:
    #     position = (testsubs[0][k], testsubs[1][k])
    #     if testlabels[k] ==0:
    #         print(codes[testsubs[0][k]], codes[testsubs[1][k]])
    # print("Ready!")

