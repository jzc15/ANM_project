import pandas as pd
from io import StringIO
from sklearn import linear_model
from numpy import *
import pylab as pl
import tqdm, os
import sys, time

sys.path.append('src/')

from lib.aq_fill import end_time

# 线性回归
def standRegres(dataSet, labelSet):
    xMat = mat(dataSet)
    yMat = mat(labelSet).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) is 0.0:
        print("Warning !!!")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws # 回归系数

# 局部加权回归
def lwlr(testPoint, xArr, yArr, k):
    xMat = mat(xArr)
    yMat = mat(yArr)
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for i in range(m):
        diffMat = testPoint - xMat[i, :]
        diffMat = diffMat/k
        weights[i, i] = exp(diffMat * diffMat.T/(-2.0))      # 计算权重对角矩阵
        #print(diffMat * diffMat.T)
    xTx = xMat.T * (weights * xMat)                                 # 奇异矩阵不能计算
    if linalg.det(xTx) == 0.0:
        print('This Matrix is singular, cannot do inverse')
        print(weights)
        return
    theta = xTx.I * (xMat.T * (weights * yMat))                     # 计算回归系数
    return testPoint * theta

# 对每个点进行高斯函数求值
def lwlrTest(xArr, yArr, k=1):
    yPre = zeros(shape(xArr)[0])
    # print "k is:", k
    len = shape(xArr)[0]
    for i in range(len):
        yPre[i] = lwlr(xArr[i], xArr, yArr, k)
    return yPre # yPre是预测值

# 打印图像
def outPic(xArr, yArr, yPre, k):
    # 线性回归
    
    #theta = theta.tolist()
    #pl.xlim(0, 3000, 100)
    #pl.ylim(0, 1000, 50)
    #pl.scatter(array(xArr[:, -1]), array(yArr), s=1, color='red', alpha=1)
    #x = arange(0, 3000, 100)
    #yHat = theta[0] + theta[1] * x  # 方程 y=ax+b
    #pl.plot(x, yHat, '-')
    #pl.show()

    # 局部加权回归
    #print(xArr.shape)
    #print(yArr.shape)
    pl.scatter(array(xArr[:, -1]), array(yArr), s=1, color='red', alpha=1)
    xArr = mat(xArr)
    srtInd = xArr[:, 1].argsort(0)
    xSort = xArr[srtInd][:, 0, :]
    
    #print(xSort[:, 1].shape)
    pl.plot(xSort[:, 1], yPre, '-')
    pl.xlabel("k = %.2f" % k)
    pl.show()
    
if __name__ == '__main__':

    # 房屋面积与价格历史数据(csv文件)

    # 读入dataframe
    
    TT = 80
    allfile = os.listdir('data/work')
    finish = []
    k = {'PM2.5': 380, 'PM10': 810, 'O3': 430}
    for file in allfile:
        if file in finish:
            continue
        #if file[0].lower() in 'abcdefmnopqrstuvwxyz':
        #    continue
        data = pd.read_csv("data/work/%s" %(file))
        print(file)
        name = file.split('.')[0].split('_')[0]
        
        ans = {}
        start = end_time('beijing') - 200
        end = end_time('beijing') + TT
        
        tylist = ['PM2.5', 'PM10']
        if(len(name) < 4):
            start = end_time('london') - 200
            end = end_time('london') + TT
        else:
            tylist.append('O3')
        for ty in tylist:
            xArr = []
            yArr = []
            ft = data['time']
            fp = data[ty]    #----
            l = len(ft)
            #start = ft.index[0]
            #print(l)
            for i in range(0, TT):
                yArr.append([])
            #print(ft)
            for i in range(0, l):
                #print(fp)
                #if(len(yArr[0]) > 500):
                #    break
                if(i >=  TT*2 and ft[i] == ft[i-TT*2] + TT*2):
                    t = [1]
                    for x in range(i-TT*2, i-TT):
                        t.append(float(fp[x]))
                    xArr.append(t)
                    #print(len(t))
                    for j in range(0, TT):
                        yArr[j].append(fp[j+i-TT])
                    #print(fp[i])
            xArr = mat(xArr)
            xMatT = xArr.T
            print(len(yArr[0]))
            cnt = 0
            now = []
            l = 0
            
            for i in tqdm.tqdm(range(start, end)):
                f = data[data.time == i]
                if(cnt >= TT):
                    end = i
                    break
                #print(f)
                #print(fp[f.index[0]])
                if(len(f.index) == 0):
                    #print(now[74])
                    #print(now[75])
                    if cnt == 0:
                        test = array([1]+ now[l-TT:])
                    #print("%d %d %d\n" %(start, end, i))
                    #print(len(test))
                    #print(test)
                    yPre = lwlr(test, xArr, mat(yArr[cnt]).T, k[ty])
                    now.append(float(yPre[0][0]))
                    print(float(yPre[0][0]))
                    cnt += 1
                else:
                    now.append(fp[f.index[0]])
                
                l += 1
            #'''
            l = len(now)
            x1 = []
            y1 = []
            for i in range(0, l):
                x1.append([1, i])
                y1.append(now[i])
                k0 = 1.5
            
            anst = lwlrTest(mat(x1), mat(y1).T, k0)
            #'''
            #print(anst)
            #outPic(mat(x1), mat(y1).T, anst, k0)
            ans[ty] = anst
            
            #break
        with open('data/result/%s'%(file), 'w') as output:
            output.write('stationId, time, PM2.5, PM10, O3\n')
            for i in range(start, end):
                if(len(name) < 4):
                    output.write("%s, %s, %s, %s, 0\n"% (name, str(i), str(ans['PM2.5'][i-start]), str(ans['PM10'][i-start])))
                else:
                    output.write("%s, %s, %s, %s, %s\n"% (name, str(i), str(ans['PM2.5'][i-start]), str(ans['PM10'][i-start]), str(ans['O3'][i-start])))
        print(time.gmtime())