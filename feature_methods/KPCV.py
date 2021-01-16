import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

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

def PC38Values(SCValues):
    Nuc2 = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]
    index = 0
    PCValues= []
    for i in range(16):
        sum = 0
        index = SCValues.index(Nuc2[i])
        for j in range(1, 39):
            sum += float(SCValues[index + j])
        sum = sum / 38
        PCValues.append(sum)
    return PCValues

def f2_kmer_pc(SSeq,PCValues):
    Nuc2 = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]
    f2_kmer_pc_feature=[]
    for seq in SSeq:
        Mat = []
        for i in range(0, len(seq) - 1 ):
            R=seq[i:i+2]
            Index = Nuc2.index(R)
            Mat.append(PCValues[Index])
        f2_kmer_pc_feature.append(Mat)
    return f2_kmer_pc_feature

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

def f2_kmer_Mat(SSeq,freq2):
    Nuc2 = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]
    index=0
    SSeq_sum=0
    f2_kmer_pc_feature=[]
    for seq in SSeq:
        Mat = np.zeros((16, 199 ))
        for i in range(0, len(seq) - 1 ):
            R=seq[i:i+2]
            Index = Nuc2.index(R)
            Mat[Index][i]=freq2[SSeq_sum][Index]
        SSeq_sum+=1
        f2_kmer_pc_feature.append(Mat)
    return f2_kmer_pc_feature

def PC12Values(SCValues):
    Nuc3=["AAA","AAC","AAG","AAT","ACA","ACC","ACG","ACT","AGA","AGC","AGG","AGT","ATA","ATC","ATG","ATT",
          "CAA","CAC","CAG","CAT","CCA","CCC","CCG","CCT","CGA","CGC","CGG","CGT","CTA","CTC","CTG","CTT",
          "GAA","GAC","GAG","GAT","GCA","GCC","GCG","GCT","GGA","GGC","GGG","GGT","GTA","GTC","GTG","GTT",
          "TAA","TAC","TAG","TAT","TCA","TCC","TCG","TCT","TGA","TGC","TGG","TGT","TTA","TTC","TTG","TTT"]
    index = 0
    PCValues= []
    for i in range(64):
        sum = 0
        index = SCValues.index(Nuc3[i])
        for j in range(1, 13):
            sum += float(SCValues[index + j])
        sum = sum / 12
        PCValues.append(sum)
    return PCValues

def f3_kmer_pc(SSeq,PCValues):
    Nuc3=["AAA","AAC","AAG","AAT","ACA","ACC","ACG","ACT","AGA","AGC","AGG","AGT","ATA","ATC","ATG","ATT",
          "CAA","CAC","CAG","CAT","CCA","CCC","CCG","CCT","CGA","CGC","CGG","CGT","CTA","CTC","CTG","CTT",
          "GAA","GAC","GAG","GAT","GCA","GCC","GCG","GCT","GGA","GGC","GGG","GGT","GTA","GTC","GTG","GTT",
          "TAA","TAC","TAG","TAT","TCA","TCC","TCG","TCT","TGA","TGC","TGG","TGT","TTA","TTC","TTG","TTT"]
    f3_kmer_pc_feature=[]
    for seq in SSeq:
        Mat = []
        for i in range(0, len(seq) - 2 ):
            R=seq[i:i+3]
            Index = Nuc3.index(R)
            Mat.append(PCValues[Index])
        f3_kmer_pc_feature.append(Mat)
    return f3_kmer_pc_feature

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
    # print(all_freq2)
    return all_freq3

def f3_kmer_Mat(SSeq,freq3):
    Nuc3=["AAA","AAC","AAG","AAT","ACA","ACC","ACG","ACT","AGA","AGC","AGG","AGT","ATA","ATC","ATG","ATT",
          "CAA","CAC","CAG","CAT","CCA","CCC","CCG","CCT","CGA","CGC","CGG","CGT","CTA","CTC","CTG","CTT",
          "GAA","GAC","GAG","GAT","GCA","GCC","GCG","GCT","GGA","GGC","GGG","GGT","GTA","GTC","GTG","GTT",
          "TAA","TAC","TAG","TAT","TCA","TCC","TCG","TCT","TGA","TGC","TGG","TGT","TTA","TTC","TTG","TTT"]
    index=0
    SSeq_sum=0
    f3_kmer_pc_feature=[]
    for seq in SSeq:
        Mat = np.zeros((64, 198 ))
        for i in range(0, len(seq) - 2 ):
            R=seq[i:i+3]
            Index = Nuc3.index(R)
            Mat[Index][i]=freq3[SSeq_sum][Index]
        SSeq_sum+=1
        f3_kmer_pc_feature.append(Mat)
    return f3_kmer_pc_feature

def feature_Mat(feature1,feature2):
    feature=[]
    m=len(feature1)
    for i in range(0,m):
        A=np.array(feature1[i])
        B=np.array(feature2[i])
        Mat=np.dot(A,B.T)
        Mat=Mat.flatten()
        feature.append(Mat)
    return feature

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

def performance(labelArr, predictArr):
    #labelArr[i] is actual value,predictArr[i] is predict value
    TN, FP, FN, TP = metrics.confusion_matrix(labelArr, predictArr).ravel()
    ACC = metrics.accuracy_score(labelArr, predictArr)
    SN = metrics.recall_score(labelArr, predictArr)
    SP = TN/(FP + TN) #Specificity = TN/N  and N = TN + FP
    MCC= matthews_corrcoef(labelArr, predictArr)
    AUC = roc_auc_score(labelArr, predictArr)
    return ACC,SN,SP,MCC,AUC

def Data_process(SSeq,valuesfilename,k):
    valueslist = getSCValues(valuesfilename)
    if(k==2):
        Freq= f2_kmer(SSeq)
        kmer_Mat =f2_kmer_Mat(SSeq, Freq)
        PC38Value=PC38Values(valueslist)
        PCM38 =f2_kmer_pc(SSeq,PC38Value)
        feature=feature_Mat(kmer_Mat,PCM38)
        Data= np.array(feature)
        return Data
    elif(k == 3):
        Freq = f3_kmer(SSeq)
        kmer_Mat = f3_kmer_Mat(SSeq, Freq)
        PC12Value = PC12Values(valueslist)
        PCM12 = f3_kmer_pc(SSeq, PC12Value)
        feature = feature_Mat(kmer_Mat, PCM12)
        Data = np.array(feature)
        return Data
    else:
        print("error1")

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


def Independence_test(Data12, Label12, Data34, Label34):
    estimator = svm.SVC(kernel='linear')
    clf = estimator.fit(Data12, Label12)
    predictArr = clf.predict(Data34)
    acc, sens, spec, mcc, auc = performance(Label34, predictArr)
    print("independent dataset test",acc, sens, spec, mcc, auc)

def parameters_select(filename1,filename2,valuesfile_D,valuesfile_T):
    best_ACC = 0.0
    i = 0
    for k in range(2, 4):
        i = i + 1
        Sseq, Label = load_data(filename1, filename2)
        if (k == 2):
            Data = Data_process(Sseq, valuesfile_D,k)
        else:
            Data = Data_process(Sseq, valuesfile_T, k)
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
    Data12=Data_process(Sseq12,valuesfile_D,best_parameter['k'])
    Data34=Data_process(Sseq34,valuesfile_D,best_parameter['k'])
    Independence_test(Data12,Label12,Data34, Label34)
elif(best_parameter['k']==3):
    Data12=Data_process(Sseq12,valuesfile_T,best_parameter['k'])
    Data34=Data_process(Sseq34,valuesfile_T,best_parameter['k'])
    Independence_test(Data12,Label12,Data34, Label34)
else:
    print("error2")