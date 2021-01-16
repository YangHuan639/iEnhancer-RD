import math
import numpy as np
from sklearn import svm
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_selection import RFE
from sklearn.preprocessing import binarize
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

#kmer
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
    return all_freq4

#PseKNC
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

#KPCV
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

def Data_process(SSeq,valuesfilename):
    valueslist = getSCValues(valuesfilename)
    #Kmer
    Kmer_Freq = f4_kmer(SSeq)
    Kmer_data = np.array(Kmer_Freq)
    # print("Kmer_data:",Kmer_data.shape)

    #PseDNC
    Freq= f2_kmer(SSeq)
    corfactor= correlation_factor_PseDNC(SSeq, valueslist, 4)
    feature = PseDNC(corfactor, Freq, 4, 0.4)
    PseDNC_Data = np.array(feature)
    # print("PseDNC_Data:",PseDNC_Data.shape)

    #KPCV
    KPCV_Freq = f2_kmer(SSeq)
    kmer_Mat = f2_kmer_Mat(SSeq, KPCV_Freq)
    PC38Value = PC38Values(valueslist)
    PCM38 = f2_kmer_pc(SSeq, PC38Value)
    feature = feature_Mat(kmer_Mat, PCM38)
    KPCV_Data = np.array(feature)
    # print("KPCV_Data:",KPCV_Data.shape)

    data_12 = np.hstack([Kmer_data, PseDNC_Data,KPCV_Data])
    # print(data_12.shape)
    return data_12

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

def evaluation(y1,y2):
    y_pred_prob = y1
    y_label = y2
    y_pred_class = binarize([y_pred_prob], 0.5)[0]

    TN, FP, FN, TP = metrics.confusion_matrix(y_label, y_pred_class).ravel()
    print(metrics.confusion_matrix(y_label, y_pred_class))
    ACC = metrics.accuracy_score(y_label, y_pred_class)
    Sens = metrics.recall_score(y_label, y_pred_class)
    Spec = TN / float(TN + FP)
    MCC = metrics.matthews_corrcoef(y_label, y_pred_class)
    AUC = metrics.roc_auc_score(y_label, y_pred_prob)
    return ACC,Sens,Spec,MCC,AUC

def creat_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def Independence_test(Data12,Label12,Data34, Label34,input_dim,batch_size,epochs):
    estimator = svm.SVC(kernel='linear')
    selector = RFE(estimator=estimator, n_features_to_select=200).fit(Data12, Label12)
    new_train_data=Data12[:, selector.support_]
    new_test_data =Data34[:, selector.support_]

    estimator = creat_model(input_dim)
    estimator.fit(new_train_data, Label12, batch_size=batch_size, epochs=epochs)
    y_pred=estimator.predict(new_test_data)
    acc, sens, spec, mcc, auc= evaluation(y_pred.flatten(), Label34)
    print("independent dataset test",acc, sens, spec, mcc, auc)

def parameters_select(Train_val,Label,input_dim):
    best_ACC = 0.0
    i = 0
    kf = StratifiedKFold(n_splits=5)
    for batch_size in [16, 32, 64]:
        for epochs in [10,50, 100]:
            i = i + 1
            print("number of times：", i)
            ACC = []
            for train, val in kf.split(Train_val, Label):
                X_train, X_val, y_train, y_val = Train_val[train], Train_val[val], Label[train], Label[val]
                estimator = svm.SVC(kernel='linear')
                selector=RFE(estimator=estimator, n_features_to_select=input_dim).fit(X_train, y_train)
                new_X_train = X_train[:, selector.support_]
                new_X_val = X_val[:, selector.support_]

                clf = creat_model(input_dim=input_dim)
                clf.fit(new_X_train, y_train, batch_size=batch_size, epochs=epochs)
                predictArr = clf.predict(new_X_val)
                acc, sens, spec, mcc, auc = evaluation(predictArr.flatten(), y_val)
                ACC.append(acc)
            ACC1 = np.array(ACC)
            ACC2 = ACC1.mean()
            print("acc", ACC2)
            if ACC2 > best_ACC:
                best_ACC = ACC2
                best_parameter = {'batch_size': batch_size, 'epochs': epochs}
    print("Best score:", best_ACC)
    print("best_parameter", best_parameter)
    return best_parameter

filename1="../B_Enhancer.txt"
filename2="../B_NonEnhancer.txt"
valuesfile_D = "../dinucleotides_revise.csv"
valuesfile_T = "../trinucleotides_revise.csv"
filename3="../I_Enhancer.txt"
filename4="../I_NonEnhancer.txt"
input_dim=200
Sseq12,Label12=load_data(filename1,filename2)
Sseq34, Label34 = load_data(filename3, filename4)
Data12=Data_process(Sseq12,valuesfile_D)
Data34=Data_process(Sseq34,valuesfile_D)
best_parameter=parameters_select(Data12,Label12,input_dim)
Independence_test(Data12,Label12,Data34, Label34,input_dim,best_parameter['batch_size'],best_parameter['epochs'])