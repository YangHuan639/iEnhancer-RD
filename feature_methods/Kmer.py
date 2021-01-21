import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
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

def f1_kmer(SSeq):
    num = 0
    all_freq1 = []
    Nuc1=[]
    for firchCh in ['A', 'C', 'G', 'T']:
        one_mer = firchCh
        Nuc1.append(one_mer)
    print(len(Nuc1))
    for seq in SSeq:
        f1 = []
        for N in Nuc1:
            for i in range(len(seq)):
                if seq[i:i+1]==N:
                    num=num+1
                else:
                    continue
            f1.append(num/200)
            num = 0
        all_freq1.append(f1)
    # print(all_freq2)
    return all_freq1

def f2_kmer(SSeq):
    num = 0
    all_freq2 = []
    Nuc2=[]
    for firchCh in ['A', 'C', 'G', 'T']:
        for secondCh in ['A', 'C', 'G', 'T']:
            two_mer = firchCh + secondCh
            Nuc2.append(two_mer)
    print(len(Nuc2))
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

def f3_kmer(SSeq):
    num = 0
    all_freq3 = []
    Nuc3=[]
    for firchCh in ['A', 'C', 'G', 'T']:
        for secondCh in ['A', 'C', 'G', 'T']:
            for thirdCh in ['A', 'C', 'G', 'T']:
                    three_mer = firchCh + secondCh + thirdCh
                    Nuc3.append(three_mer)
    print(len(Nuc3))
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

def f4_kmer(SSeq):
    num = 0
    all_freq4 = []
    Nuc4=[]
    for firchCh in ['A', 'C', 'G', 'T']:
        for secondCh in ['A', 'C', 'G', 'T']:
            for thirdCh in ['A', 'C', 'G', 'T']:
                for fourthCh in ['A', 'C', 'G', 'T']:
                    four_mer = firchCh + secondCh + thirdCh + fourthCh
                    Nuc4.append(four_mer)
    print(len(Nuc4))
    for seq in SSeq:
        f4 = []
        for N in Nuc4:
            for i in range(len(seq) - 3):
                if seq[i:i+4]==N:
                    num=num+1
                else:
                    continue
            f4.append(num/197)
            num = 0
        all_freq4.append(f4)
    # print(all_freq2)
    return all_freq4

def f5_kmer(SSeq):
    num = 0
    all_freq5 = []
    Nuc5 = []
    for firchCh in ['A', 'C', 'G', 'T']:
        for secondCh in ['A', 'C', 'G', 'T']:
            for thirdCh in ['A', 'C', 'G', 'T']:
                for fourthCh in ['A', 'C', 'G', 'T']:
                    for fifthCh in ['A', 'C', 'G', 'T']:
                        five_mer = firchCh + secondCh + fourthCh + thirdCh + fifthCh
                        Nuc5.append(five_mer)
    print(len(Nuc5))
    for seq in SSeq:
        f5 = []
        for N in Nuc5:
            for i in range(len(seq) - 4):
                if seq[i:i + 5] == N:
                    num = num + 1
                else:
                    continue
            f5.append(num / 196)
            num = 0
        all_freq5.append(f5)
    # print(all_freq2)
    return all_freq5

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

def Data_process(SSeq,k):
    if(k==1):
        Freq = f1_kmer(SSeq)
        data_np = np.array(Freq)
        return data_np
    elif(k==2):
        Freq = f2_kmer(SSeq)
        data_np = np.array(Freq)
        return data_np
    elif(k==3):
        Freq = f3_kmer(SSeq)
        data_np = np.array(Freq)
        return data_np
    elif(k==4):
        Freq = f4_kmer(SSeq)
        data_np = np.array(Freq)
        return data_np
    elif(k==5):
        Freq = f5_kmer(SSeq)
        data_np = np.array(Freq)
        return data_np
    else:
        print("error")

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

def performance(labelArr, predictArr):
    #labelArr[i] is actual value,predictArr[i] is predict value
    TN, FP, FN, TP = metrics.confusion_matrix(labelArr, predictArr).ravel()
    ACC = metrics.accuracy_score(labelArr, predictArr)
    SN = metrics.recall_score(labelArr, predictArr)
    SP = TN/(FP + TN) #Specificity = TN/N  and N = TN + FP
    MCC= matthews_corrcoef(labelArr, predictArr)
    return ACC,SN,SP,MCC

def Independence_test(Data12, Label12, Data34, Label34):
    estimator = svm.SVC(kernel='linear')
    clf = estimator.fit(Data12, Label12)
    predictArr = clf.predict(Data34)
    acc, sens, spec, mcc= performance(Label34, predictArr)
    print("independent dataset test",acc, sens, spec, mcc, auc)

def parameters_select(filename1,filename2):
    best_ACC = 0.0
    i = 0
    for k in range(1, 6):
        i = i + 1
        Sseq, Label = load_data(filename1, filename2)
        Train_val = Data_process(Sseq, k)
        kf = StratifiedKFold(n_splits=5)
        ACC = []
        for train, val in kf.split(Train_val, Label):
            X_train, X_val, y_train, y_val = Train_val[train], Train_val[val], Label[train], Label[val]
            estimator = svm.SVC(kernel='linear')
            clf = estimator.fit(X_train, y_train)
            predictArr = clf.predict(X_val)
            acc, sens, spec, mcc, auc = performance(y_val, predictArr)
            ACC.append(acc)
        ACC = np.array(ACC).mean()
        print("number of times{}".format(i))
        print("acc", ACC)
        if ACC > best_ACC:
            best_ACC = ACC
            best_parameter = {'k': k}
    print("Best score:",best_ACC)
    print("Best parameters:{}".format(best_parameter))
    return best_parameter

#data:layer Ⅰ
filename1="../B_Enhancer.txt"
filename2="../B_NonEnhancer.txt"
filename3="../I_Enhancer.txt"
filename4="../I_NonEnhancer.txt"

"""
#data:layer Ⅱ
filename1="../B_StrongEnhancer.txt"
filename2="../B_WeakEnhancer.txt"
filename3="../I_StrongEnhancer.txt"
filename4="../I_WeakEnhancer.txt"
"""

best_parameter=parameters_select(filename1,filename2)

Sseq12,Label12=load_data(filename1,filename2)
Sseq34, Label34 = load_data(filename3, filename4)
Data12=Data_process(Sseq12,best_parameter['k'])
Data34=Data_process(Sseq34,best_parameter['k'])
Independence_test(Data12,Label12,Data34, Label34)
