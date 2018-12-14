from iops.evaluation.evaluation import get_range_proba
import os

def get_data(inputfile):

    inputdata = open(inputfile, 'r')
    key = []
    value = []
    time0 = None
    for line in inputdata:
        line = line.strip()
        lines = line.split(' ')
        key.append(float(lines[0]))
        value.append(int(lines[1]))
    return key, value

def get_f_score(TP, TN, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return precision * recall * 2 / (precision + recall)

if __name__ == "__main__":
    files = os.listdir('result_vae/')
    for case in files:
        kl = 0
        time0 = None
        tdt = None

        key, lable = get_data("result_vae/" + case + "/test")   
        key = get_range_proba(key, lable)
        print(key)
        exit(0)
        
        task = []
        for i, x in enumerate(key):
            task.append((key[i], value[i]))
        
        task.sort(key = lambda k:k[0])
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0


        for x in task:
            if x[1] == 1:
                FP += 1
            else:
                TP += 1
        
        ans = task[0][0]
        f = get_f_score(TP, TN, FP, FN)

        for x in task:
            if x[1] == 1:
                FP -= 1
                TN += 1
            else:
                TP -= 1
                FN += 1
            f0 = get_f_score(TP, TN, FP, FN)
            if f0 > f:
                ans = x[0]
    
        print(ans, f)
        break