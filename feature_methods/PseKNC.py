import math
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def getDNA(filename):
    fr = open(filename, 'r')
    SSeq=[]
    for line in fr.readlines():
        line = line.strip('\n')
        if(line[0]!='>'):
            str=line.upper()
            SSeq.append(str)
    return SSeq

def getSCValues(valuesfile):
    valueslist=[]
    fr = open(valuesfile, 'r')
    for lines in fr.readlines():
        lines = lines.strip('\n')
        lines=lines.split(",")
        for line in lines:
            valueslist.append(line)
    return valueslist

def f2_kmer(SSeq):
    Nuc2=["AA","AC","AG","AT","CA","CC","CG","CT","GA","GC","GG","GT","TA","TC","TG","TT"]
    num = 0
    all_freq2 = []
    for seq in SSeq:
        f2 = []
        for N in Nuc2:
            for i in range(len(seq) - 1):
                if seq[i:i+2]==N:
                    num=num+1
                else:
                    continue
            f2.append(num/199)
            num = 0
        all_freq2.append(f2)
    # print(all_freq2)
    return all_freq2

def correlation_factor_PseDNC(SSeq,SCValues,iter):
    list=[]
    for seq in SSeq:
        Theta = []
        for j in range(1, iter + 1):
            SUM = []
            for i in range(0, len(seq)-j-1):
                R1 = seq[i:i+2]
                R2 = seq[i+j:i+j+2]
                index1 = SCValues.index(R1)
                index2 = SCValues.index(R2)
                sum=0
                for z in range(1, 39):
                    PC1=float(SCValues[index1 + z])
                    PC2=float(SCValues[index2 + z])
                    sum+=math.pow((PC1-PC2),2)
                sum=sum/38
                SUM.append(sum)
            SUM=np.array(SUM)
            theta=np.sum(SUM)/(199-j)
            Theta.append(theta)
        list.append(Theta)
    return list

def PseDNC(corfactor,freq,iter,w):
    Nuc2 = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]
    list=[]
    Vetor = []
    for fr,cor in zip(freq,corfactor):
        vet = []
        fr = np.array(fr)
        Freq = np.sum(fr)
        cor = np.array(cor)
        Cor = np.sum(cor)
        for i in range(len(Nuc2)):
            A = fr[i]
            B = float(Freq) + float(w * float(Cor))
            Du1 = A / B
            vet.append(Du1)
        for j in range(iter):
            A = w * float(cor[j])
            B = float(Freq) + float(w * float(Cor))
            Du2 = A / B
            vet.append(Du2)
        Vetor.append(vet)
    return Vetor

def f3_kmer(SSeq):
    Nuc3=["AAA","AAC","AAG","AAT","ACA","ACC","ACG","ACT","AGA","AGC","AGG","AGT","ATA","ATC","ATG","ATT",
          "CAA","CAC","CAG","CAT","CCA","CCC","CCG","CCT","CGA","CGC","CGG","CGT","CTA","CTC","CTG","CTT",
          "GAA","GAC","GAG","GAT","GCA","GCC","GCG","GCT","GGA","GGC","GGG","GGT","GTA","GTC","GTG","GTT",
          "TAA","TAC","TAG","TAT","TCA","TCC","TCG","TCT","TGA","TGC","TGG","TGT","TTA","TTC","TTG","TTT"]
    num = 0
    all_freq3 = []
    for seq in SSeq:
        f3 = []
        for N in Nuc3:
            for i in range(len(seq) - 2):
                if seq[i:i+3]==N:
                    num=num+1
                else:
                    continue
            f3.append(num/198)
            num = 0
        all_freq3.append(f3)
    return all_freq3

def correlation_factor_PseTNC(SSeq,SCValues,iter):
    list=[]
    for seq in SSeq:
        Theta = []
        for j in range(1, iter + 1):
            SUM = []
            for i in range(0, len(seq)-j-2):
                R1 = seq[i:i+3]
                R2 = seq[i+j:i+j+3]
                index1 = SCValues.index(R1)
                index2 = SCValues.index(R2)
                sum=0
                for z in range(1, 13):
                    PC1=float(SCValues[index1 + z])
                    PC2=float(SCValues[index2 + z])
                    sum+=math.pow((PC1-PC2),2)
                sum=sum/12
                SUM.append(sum)
            SUM=np.array(SUM)
            theta=np.sum(SUM)/(198-j)
            # print("ith theta",i,theta)
            Theta.append(theta)
        # print("jth Theta",j,Theta)
        list.append(Theta)
    # print("list.shape",np.array(list).shape)
    # print(list)
    return list

def PseTNC(corfactor,freq,iter,w):
    Nuc3=["AAA","AAC","AAG","AAT","ACA","ACC","ACG","ACT","AGA","AGC","AGG","AGT","ATA","ATC","ATG","ATT",
          "CAA","CAC","CAG","CAT","CCA","CCC","CCG","CCT","CGA","CGC","CGG","CGT","CTA","CTC","CTG","CTT",
          "GAA","GAC","GAG","GAT","GCA","GCC","GCG","GCT","GGA","GGC","GGG","GGT","GTA","GTC","GTG","GTT",
          "TAA","TAC","TAG","TAT","TCA","TCC","TCG","TCT","TGA","TGC","TGG","TGT","TTA","TTC","TTG","TTT"]
    Vetor = []
    for fr,cor in zip(freq,corfactor):
        vet = []
        fr = np.array(fr)
        Freq = np.sum(fr)
        cor = np.array(cor)
        Cor = np.sum(cor)
        for i in range(len(Nuc3)):
            A = fr[i]
            B = float(Freq) + float(w * float(Cor))
            Du1 = A / B
            vet.append(Du1)
        for j in range(iter):
            A = w * float(cor[j])
            B = float(Freq) + float(w * float(Cor))
            Du2 = A / B
            vet.append(Du2)
        Vetor.append(vet)
    return Vetor

def label_Pos(feature):
    label_Pos=[]
    for i in range(len(feature)):
        label_Pos.append(1)
    return label_Pos

def label_Neg(feature):
    label_Neg=[]
    for i in range(len(feature)):
        label_Neg.append(0)
    return label_Neg

def load_data(filename1, filename2):
    Dnaseq1 = getDNA(filename1)
    Dnaseq2 = getDNA(filename2)

    data1_np = np.array(Dnaseq1)
    data2_np = np.array(Dnaseq2)
    data_12 = np.hstack([data1_np, data2_np])

    y1 = label_Pos(Dnaseq1)
    y2 = label_Neg(Dnaseq2)
    y1_np = np.array(y1)
    y2_np = np.array(y2)
    y12 = np.hstack([y1_np, y2_np])
    return data_12,y12

def Data_process_PseDNC(SSeq,valuesfilename,iter,w):
    Freq= f2_kmer(SSeq)
    valueslist = getSCValues(valuesfilename)
    corfactor= correlation_factor_PseDNC(SSeq, valueslist, iter)
    feature = PseDNC(corfactor, Freq, iter, w)

    Data= np.array(feature)

    return Data

def Data_process_PseTNC(SSeq,valuesfilename,iter,w):
    Freq= f3_kmer(SSeq)
    valueslist = getSCValues(valuesfilename)
    corfactor= correlation_factor_PseTNC(SSeq, valueslist, iter)
    feature = PseTNC(corfactor, Freq, iter, w)

    data_feature = np.array(feature)

    return data_feature

def performance(labelArr, predictArr):
    #labelArr[i] is actual value,predictArr[i] is predict value
    TN, FP, FN, TP = metrics.confusion_matrix(labelArr, predictArr).ravel()
    ACC = metrics.accuracy_score(labelArr, predictArr)
    SN = metrics.recall_score(labelArr, predictArr)
    SP = TN/(FP + TN) #Specificity = TN/N  and N = TN + FP
    MCC= matthews_corrcoef(labelArr, predictArr)
    AUC = roc_auc_score(labelArr, predictArr)
    return ACC,SN,SP,MCC,AUC

def Independence_test(Data12, Label12, Data34, Label34):
    estimator = svm.SVC(kernel='linear')
    clf = estimator.fit(Data12, Label12)
    predictArr = clf.predict(Data34)
    acc, sens, spec, mcc, auc= performance(Label34, predictArr)
    print("independent dataset test",acc, sens, spec, mcc, auc)

def parameters_select(filename1,filename2,valuesfile_D,valuesfile_T):
    best_ACC = 0.0
    i = 0
    for k in [2, 3]:
        for iter in range(1, 21):
            for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:  # omega
                i = i + 1
                Sseq, Label = load_data(filename1, filename2)
                if (k == 2):
                    Data = Data_process_PseDNC(Sseq, valuesfile_D, iter, w)
                else:
                    Data = Data_process_PseTNC(Sseq, valuesfile_T, iter, w)
                kf = StratifiedKFold(n_splits=5)
                ACC = []
                for train, val in kf.split(Data, Label):
                    X_train, X_val, y_train, y_val = Data[train], Data[val], Label[train], Label[val]
                    estimator = svm.SVC(kernel='linear')
                    clf = estimator.fit(X_train, y_train)
                    predictArr = clf.predict(X_val)
                    acc, sens, spec, mcc, auc = performance(y_val, predictArr)
                    ACC.append(acc)
                ACC = np.array(ACC).mean()
                print("number of times{}".format(i))
                print("acc",ACC)
                if ACC > best_ACC:
                    best_ACC = ACC
                    best_parameter = {'k':k, 'iter': iter,'w': w}
    print("Best score:",best_ACC)
    print("Best parameters:{}".format(best_parameter))
    return best_parameter

#data:layer Ⅰ
filename1="../B_Enhancer.txt"
filename2="../B_NonEnhancer.txt"
valuesfile_D = "../dinucleotides_revise.csv"
valuesfile_T = "../trinucleotides_revise.csv"
filename3="../I_Enhancer.txt"
filename4="../I_NonEnhancer.txt"

"""
#data:layer Ⅱ
filename1="../B_StrongEnhancer.txt"
filename2="../B_WeakEnhancer.txt"
filename3="../I_StrongEnhancer.txt"
filename4="../I_WeakEnhancer.txt"
"""

best_parameter=parameters_select(filename1,filename2,valuesfile_D,valuesfile_T)
Sseq12,Label12=load_data(filename1,filename2)
Sseq34, Label34 = load_data(filename3, filename4)

if(best_parameter['k']==2):
    Data12 = Data_process_PseDNC(Sseq12, valuesfile_D, best_parameter['iter'], best_parameter['w'])
    Data34 = Data_process_PseDNC(Sseq34, valuesfile_D, best_parameter['iter'], best_parameter['w'])
    Independence_test(Data12,Label12,Data34, Label34)
elif(best_parameter['k']==3):
    Data12 = Data_process_PseTNC(Sseq12, valuesfile_T, best_parameter['iter'], best_parameter['w'])
    Data34 = Data_process_PseTNC(Sseq34, valuesfile_T, best_parameter['iter'], best_parameter['w'])
    Independence_test(Data12, Label12, Data34, Label34)
else:
    print("error2")