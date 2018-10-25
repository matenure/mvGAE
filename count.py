import numpy as np

if __name__ == '__main__':
    resPath = "MLK_50.txt"
    with open(resPath,"rb") as f:
        listMethod = []
        dictValues = dict()
        for line in f:
            if len(line.strip())==0:
                continue
            tabs = line.strip().split()

            listMethod.append(tabs[0])
            if tabs[0] not in dictValues:
                dictValues[tabs[0]] = [[float(tabs[1]), float(tabs[2])]]
            dictValues[tabs[0]].append([float(tabs[1]), float(tabs[2])])
    wf = open("res_MLK_50.txt","wb")
    for method in dictValues.keys():
        resMat = np.array(dictValues[method])
        rmean = np.mean(resMat,0)
        rstd = np.std(resMat,0)
        wf.write(method+"\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(rmean[0],rstd[0],rmean[1],rstd[1]))
    wf.close()