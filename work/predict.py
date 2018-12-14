from copy import deepcopy
from numpy import *
import random  # , tqdm
import os

N = 100
th = 0.2
kl = None
time0 = None
tdt = None
# 线性回归


def standRegres(dataSet, labelSet):
    xMat = mat(dataSet)
    yMat = mat(labelSet).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) is 0.0:
        print("Warning !!!")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws  # 回归系数

# 局部加权回归


def lwlr(testPoint, xArr, yArr, k):
    xMat = mat(xArr)
    yMat = mat(yArr)
    m = shape(xMat)[0]
    # print(m)
    weights = mat(eye((m)))
    for i in range(m):
        diffMat = testPoint - xMat[i, :]
        diffMat = diffMat / k
        weights[i, i] = exp(diffMat * diffMat.T / (-2.0))      # 计算权重对角矩阵
        # print(diffMat * diffMat.T)
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
    return yPre  # yPre是预测值

# 打印图像


def outPic(xArr, yArr, yPre, k):
    # 线性回归

    # theta = theta.tolist()
    # pl.xlim(0, 3000, 100)
    # pl.ylim(0, 1000, 50)
    # pl.scatter(array(xArr[:, -1]), array(yArr), s=1, color='red', alpha=1)
    # x = arange(0, 3000, 100)
    # yHat = theta[0] + theta[1] * x  # 方程 y=ax+b
    # pl.plot(x, yHat, '-')
    # pl.show()

    # 局部加权回归
    # print(xArr.shape)
    # print(yArr.shape)
    pl.scatter(array(xArr[:, -1]), array(yArr), s=1, color='red', alpha=1)
    xArr = mat(xArr)
    srtInd = xArr[:, 1].argsort(0)
    xSort = xArr[srtInd][:, 0, :]

    # print(xSort[:, 1].shape)
    pl.plot(xSort[:, 1], yPre, '-')
    pl.xlabel("k = %.2f" % k)
    pl.show()


def get_test(inputfile):
    global tdt
    global kl
    inputdata = open(inputfile, 'r')
    t = []
    time0 = None
    for line in inputdata:
        line = line.strip()
        keys = line.split(' ')
        time = int(keys[0])
        value = float(keys[1])
        if kl <= 5:
            value += 5
        if time0 == None:
            time0 = time
        times = (time - time0) // tdt
        t.append((times, value, time))
    return t


def get_data(inputfile):
    global kl
    global time0
    global tdt
    inputdata = open(inputfile, 'r')
    t = []
    time0 = None
    for line in inputdata:
        line = line.strip()
        keys = line.split(' ')
        time = int(keys[0])
        value = float(keys[1])
        lable = int(keys[2])
        if time0 == None:
            time0 = time
        if tdt == 1:
            tdt = time - time0
        if kl == 0:
            tdt = 1
        if lable == 0:
            kl = max(kl, value)
            kl = max(kl, -value)
        times = (time - time0) // tdt
        t.append((times, value, lable))
    return t


def check(u, v):
    dt = max(u - v, v - u)
    return dt / u < th

if __name__ == "__main__":
    files = os.listdir('../train/')
    flist = ['7c189dd36f048a6c', '18fbb1d5a5dc099d', 'da403e4e3f87c9e0', '1c35dbf57f55f5e4']
    for case in files:
        try:
            kl = 0
            time0 = None
            tdt = None
            print(case)
            t = get_data("../train/" + case + "/train")
            z = []
            xArr = []
            yArr = []
            dt = 0
            for i, x in enumerate(t):

                if x[2] == 1 or x[0] - i != dt:
                    z = []
                    dt = x[0] - i
                xx = x[1]
                z.append(xx + 5)
                if len(z) == N + 1:
                    a = deepcopy(z[:N])
                    b = deepcopy(z[N])
                    xArr.append(a)
                    yArr.append(b)
                    z = z[1:]

            l = len(yArr)
            print(time0)
            print(tdt)
            axArr = []
            bxArr = []

            li = []
            for i in range(0, 300):
                while True:
                    x = random.randint(0, l - 1)
                    if x not in li:
                        break
                # print(x)
                li.append(x)
                axArr.append(xArr[x])
                bxArr.append(yArr[x])
            xArr = mat(axArr)
            yArr = mat(bxArr).T

            test = get_test("../test/" + case + "/test")
            z = []
            ct = N
            total = len(test)
            total = 200
            totalt = test[total - 1][0]

            kl += 5

            for x in range(0, N):
                z.append(test[x][1])
                test[x] = list(test[x])
                test[x].append(0)
            for x in range(N, totalt + 1):
                # print(x)
                z_array = array(z)

                # print(z)
                p = lwlr(z_array, xArr, yArr, kl)
                p = float(p[0][0])
                if x == test[ct][0]:
                    if check(p, test[ct][1]):
                        p = test[ct][1]
                        test[ct] = list(test[ct])
                        test[ct].append(0)
                    else:
                        test[ct] = list(test[ct])
                        test[ct].append(1)
                    ct += 1

                z = z[1:]
                z.append(p)
            text = "case,timestamp,predict\n"

            for x in range(0, total):
                text += case + ',' + \
                    str(test[x][2]) + ',' + str(test[x][3]) + '\n'
            output = open(case + '_result.csv', 'w')
            output.write(text)
        except:
            print(case, 'error')
            test = get_test("../test/" + case + "/test")
            total = len(test)
            text = "case,timestamp,predict\n"
            for x in range(0, total):
                th = 1
                if kl <= 5:
                    th += 5
                if (test[x][1] > th):
                    text += case + ',' + \
                        str(test[x][2]) + ',' + '1' + '\n'
                else:
                    text += case + ',' + \
                        str(test[x][2]) + ',' + '0' + '\n'
            output = open(case + '_result.csv', 'w')
            output.write(text)
