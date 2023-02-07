#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# chaos_prediction.py: 本書の図4.9に対応するサンプルコード
#################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from quaternion_calculation import *
from qmodel import ESN, QGC
import os


np.random.seed(seed=0)

# Lorenz方程式によるデータ生成
# class Lorenz:
#     # パラメータの指定
#     def __init__(self, sigma, r, b):
#         self.sigma = sigma
#         self.r = r
#         self.b = b

#     def f1(self, t, x, y, z):
#         return -self.sigma*x + self.sigma*y

#     def f2(self, t, x, y, z):
#         return -x*z + self.r*x - y

#     def f3(self, t, x, y, z):
#         return x*y - self.b*z

#     def Lorenz(self, t, X):
#         '''
#         :param t: 時間
#         :param X: 3次元ベクトル
#         :return: 3次元ベクトル
#         '''
#         next_X = [self.f1(t, X[0], X[1], X[2]), 
#                   self.f2(t, X[0], X[1], X[2]), 
#                   self.f3(t, X[0], X[1], X[2])]
#         return np.array(next_X)

#     # 4次のRunge-Kutta法により数値積分
#     def Runge_Kutta(self, x0, T, dt):
#         '''
#         :param x0: 初期値
#         :param T: 時間
#         :param dt: ステップ幅
#         :return: Lorenz方程式の時系列, (T/dt)x3
#         '''
#         X = x0
#         t = 0
#         data = []

#         while t < T:
#             k_1 = self.Lorenz(t, X)
#             k_2 = self.Lorenz(t + dt/2, X + dt*k_1/2)
#             k_3 = self.Lorenz(t + dt/2, X + dt*k_2/2)
#             k_4 = self.Lorenz(t + dt, X + dt*k_3)
#             next_X = X + dt/6*(k_1 + 2*k_2 + 2*k_3 + k_4)
#             data.append(next_X)
#             X = next_X
#             t = t + dt

#         return np.array(data)
    
    # ３次元データを四元数
def to_quaternion(data):
    data_len = len(data)
    qdata = np.full(data_len, Variable(np.array([0,0,0,0])))
    for i in range(data_len):
        tmp = np.insert(data[i], 0, 0)
        qdata[i] = Variable(tmp)
    return qdata

def to_3Ddata(qdata):
    data_len = len(qdata)
    data = np.full((data_len, 3), np.array([0, 0, 0]))
    for i in range(data_len):
        for j in range(3):
            data[i][j] = qdata[i].data[j+1]
    return data

if __name__ == '__main__':
    f = "data/20230207_class_3.dat"
    df = np.genfromtxt(os.path.join(f), delimiter=",")
    df = (df-2048)/2048*100 #DA変換 [mV]

    #N = df.shape[0]           # サンプル数
    N=600
    freq = 20 # 周波数
    tmax = N/freq #計測時間
    # データのパラメータ

    dt = 1/freq          # サンプリング間隔
    fc = 2  # カットオフ周波数
    t = np.arange(0, N*dt, dt)  # 時間軸
    fq = np.linspace(0, 1.0/dt, N//2)  # 周波数軸

    # Tx1-Rx1 Svv=I1+jQ1
    I1 = df[0:N:2, 0]
    Q1 = df[0:N:2, 1]
    #A1 = np.sqrt((I1 **2 + Q1 ** 2))
    #P1=(np.arctan2(Q1,I1))
    # Tx1-Rx2 Svh=I2+jQ2
    I2 = df[0:N:2, 2]
    Q2 = df[0:N:2, 3]
    #A2 = np.sqrt((I2 ** 2 + Q2 ** 2))
    #P2=(np.arctan2(Q2,I2))
    # Tx2-Rx1 Shv=I3+jQ3
    I3 = df[1:N:2, 0]
    Q3 = df[1:N:2, 1]
    #A3 = np.sqrt((I3 ** 2 + Q3 ** 2))
    #P3=(np.arctan2(Q3,I3))
    # Tx2-Rx2 Shh=I4+jQ4
    I4 = df[1:N:2, 2]
    Q4 = df[1:N:2, 3]
    #A4 = np.sqrt((I4 ** 2 + Q4 ** 2))
    #P4=(np.arctan2(Q4,I4))

    # 高速フーリエ変換（周波数信号に変換）
    F1 = np.fft.fft(I1)
    F2 = np.fft.fft(I2)
    F3 = np.fft.fft(I3)
    F4 = np.fft.fft(I4)
    G1 = np.fft.fft(Q1) 
    G2 = np.fft.fft(Q2)
    G3 = np.fft.fft(Q3)
    G4 = np.fft.fft(Q4)

    # 正規化 + 交流成分2倍
    F1 = F1/(N/2/2)
    F1[0] = F1[0]/2
    F2 = F2/(N/2/2)
    F2[0] = F2[0]/2
    F3 = F3/(N/2/2)
    F3[0] = F3[0]/2
    F4 = F4/(N/2/2)
    F4[0] = F4[0]/2
    G1 = G1/(N/2/2)
    G1[0] = G1[0]/2
    G2 = G2/(N/2/2)
    G2[0] = G2[0]/2
    G3 = G3/(N/2/2)
    G3[0] = G3[0]/2
    G4 = G4/(N/2/2)
    G4[0] = G4[0]/2

    # 配列Fをコピー
    FF1 = F1.copy()
    FF2 = F2.copy()
    FF3 = F3.copy()
    FF4 = F4.copy()
    GG1 = G1.copy()
    GG2 = G2.copy()
    GG3 = G3.copy()
    GG4 = G4.copy()

    print(FF1.shape)
    # ローパスフィル処理（カットオフ周波数を超える帯域の周波数信号を0にする)
    FF1[(fq > fc)] = 0
    FF2[(fq > fc)] = 0
    FF3[(fq > fc)] = 0
    FF4[(fq > fc)] = 0
    GG1[(fq > fc)] = 0
    GG2[(fq > fc)] = 0
    GG3[(fq > fc)] = 0
    GG4[(fq > fc)] = 0
    #F2[(freq == 0)] = 0 

    # 高速逆フーリエ変換（時間信号に戻す）
    f1 = np.fft.ifft(FF1)
    f2 = np.fft.ifft(FF2)
    f3 = np.fft.ifft(FF3)
    f4 = np.fft.ifft(FF4)
    g1 = np.fft.ifft(GG1)
    g2 = np.fft.ifft(GG2)
    g3 = np.fft.ifft(GG3)
    g4 = np.fft.ifft(GG4)

    # 振幅を元のスケールに戻す
    f1 = np.real(f1*N)
    f2 = np.real(f2*N)
    f3 = np.real(f3*N)
    f4 = np.real(f4*N)
    g1 = np.real(g1*N)
    g2 = np.real(g2*N)
    g3 = np.real(g3*N)
    g4 = np.real(g4*N)

    I1 = f1 - np.mean(f1)
    I2 = f2 - np.mean(f2)
    I3 = f3 - np.mean(f3)
    I4 = f4 - np.mean(f4)
    Q1 = g1 - np.mean(g1)
    Q2 = g2 - np.mean(g2)
    Q3 = g3 - np.mean(g3)
    Q4 = g4 - np.mean(g4)

    A1 = np.sqrt((I1 **2 + Q1 ** 2))
    P1=(np.arctan2(Q1,I1))
    A2 = np.sqrt((I2 ** 2 + Q2 ** 2))
    P2=(np.arctan2(Q2,I2))
    A3 = np.sqrt((I3 ** 2 + Q3 ** 2))
    P3=(np.arctan2(Q3,I3))
    A4 = np.sqrt((I4 ** 2 + Q4 ** 2))
    P4=(np.arctan2(Q4,I4))
    #送信-水平偏波
    g0 = (A4**2)+(A3**2) # |Shh|^2+|Svh|^2
    g1 = (A4**2)-(A3**2) # |Shh|^2-|Svh|^2
    g2= 2*(I3*I4+Q3*Q4)  # 2Re(Shh`*Svh)
    g3= 2*(I4*Q3-I3*Q4)  # 2Im(Shh`*Svh)
    #送信-垂直偏波
    h0 = (A2**2)+(A1**2)
    h1 = (A2**2)-(A1**2)
    h2= 2*(I1*I2+Q1*Q2)
    h3= 2*(I2*Q1-I1*Q2)
    #送信‐45度偏波
    m0 = 1/2*(((I2+I4)**2+(Q2+Q4)**2)+((I1+I3)**2+(Q1+Q3)**2))
    m1 = 1/2*(((I2+I4)**2+(Q2+Q4)**2)-((I1+I3)**2+(Q1+Q3)**2))
    m2 = 2*1/2*((I1+I3)*(I2+I4)+(Q1+Q3)*(Q2+Q4))
    m3 = 2*1/2*((Q1+Q3)*(I2+I4)-(I1+I3)*(Q2+Q4))
    #送信‐‐45度偏波
    n0 = 1/2*(((I2-I4)**2+(Q2-Q4)**2)+((I1-I3)**2+(Q1-Q3)**2))
    n1 = 1/2*(((I2-I4)**2+(Q2-Q4)**2)-((I1-I3)**2+(Q1-Q3)**2))
    n2 = 2*1/2*((I1-I3)*(I2-I4)+(Q1-Q3)*(Q2-Q4))
    n3 = 2*1/2*((Q1-Q3)*(I2-I4)-(I1-I3)*(Q2-Q4))
    #送信-左円偏波
    l0 = 1/2*(((I4-Q2)**2+(Q4+I2)**2)+((I3-Q1)**2+(Q3+I1)**2))
    l1 = 1/2*(((I4-Q2)**2+(Q4+I2)**2)-((I3-Q1)**2+(Q3+I1)**2))
    l2 = 2*1/2*((I3-Q1)*(I4-Q2)+(Q3+I1)*(Q4+I2))
    l3 = 2*1/2*((Q3+I1)*(I4-Q2)-(I3-Q1)*(Q4+I2))
    #送信-右円偏波
    r0 = 1/2*(((I2-Q4)**2+(Q2+I4)**2)+((I1-Q3)**2+(Q1+I3)**2))
    r1 = 1/2*(((I2-Q4)**2+(Q2+I4)**2)-((I1-Q3)**2+(Q1+I3)**2))
    r2 = 2*1/2*((I1-Q3)*(I2-Q4)+(Q1+I3)*(Q2+I4))
    r3 = 2*1/2*((Q1+I3)*(I2-Q4)-(I1-Q3)*(Q2+I4))

    T_train = 150  # 学習データの長さ
    T_test = 150  # テストデータの長さ

    qdata = np.array([
    to_quaternion(np.array([g1/g0,g2/g0,g3/g0]).T),to_quaternion(np.array([h1/h0,h2/h0,h3/h0]).T),
    to_quaternion(np.array([m1/m0,m2/m0,m3/m0]).T),to_quaternion(np.array([n1/n0,n2/n0,n3/n0]).T),
    to_quaternion(np.array([l1/l0,l2/l0,l3/l0]).T),to_quaternion(np.array([r1/r0,r2/r0,r3/r0]).T)
    ]).T

    QD1 = np.array([
    to_quaternion([[1,-1,-1]]),to_quaternion([[-1,-1,-1]]),
    to_quaternion([[-1,-1,-1]]),to_quaternion([[-1,-1,-1]]),
    to_quaternion([[-1,-1,-1]]),to_quaternion([[-1,-1,-1]])
    ]*50)
    QD2 = np.array([
    to_quaternion([[-1,-1,-1]]),to_quaternion([[-1,1,-1]]),
    to_quaternion([[-1,-1,-1]]),to_quaternion([[-1,-1,-1]]),
    to_quaternion([[-1,-1,-1]]),to_quaternion([[-1,-1,-1]])
    ]*50)
    QD3 = np.array([
    to_quaternion([[-1,-1,-1]]),to_quaternion([[-1,-1,-1]]),
    to_quaternion([[1,-1,-1]]),to_quaternion([[-1,-1,-1]]),
    to_quaternion([[-1,-1,-1]]),to_quaternion([[-1,-1,-1]])
    ]*50)

    QD1 =QD1.reshape(50,6)
    QD2 =QD2.reshape(50,6)
    QD3 =QD3.reshape(50,6)
    QD = np.concatenate([QD1, QD2, QD3])

    print(QD.shape)

    # 訓練・検証用情報
    # train_U = data[:int(T_train/dt)]
    train_QU = qdata[:T_train]
    # train_D = data[1:int(T_train/dt)+1]
    train_QD = QD

    # test_U = data[int(T_train/dt):int((T_train+T_test)/dt)]
    test_QU = qdata[T_train:]
    # test_D = data[1+int(T_train/dt):int((T_train+T_test)/dt)+1]
    train_QD = QD


    # ESNモデル
    N_x = 100  # リザバーのノード数
    rho = 0.01
    model = ESN(1, 1, N_x, \
                density=0.1, input_scale=0.1, rho=rho)

    # オンライン学習と予測
    # print(type(train_QU[0]),train_QU.shape)
    # print(train_QD.shape)
    qtrain_Y = model.qadapt(train_QU, train_QD, QGC(N_x, 0.01))
    # print(type(qtrain_Y))
    train_Y = to_3Ddata(qtrain_Y)
    # print(train_Y)
    qtest_Y = model.qfree_run(test_QU)
    test_Y = to_3Ddata(qtest_Y)
    print(test_Y,test_Y.shape)

    y = []
    div1 = []
    div2= []
    div3 = []
    E1= []
    E2 = []
    E3 = []
    P = []
    P1 = []
    P2 = []
    P3 = []
    for i in range(T_test):
        y[i] = qtest_Y[i::150]
        div1[i] = QD1[0]-y[i]
        div2[i] = QD2[0]-y[i]
        div3[i] = QD3[0]-y[i]
        E1[i] = 0
        E2[i] = 0
        E3[i] = 0
        for j in range(6):
            E1[i] += absolute_value(div1[i][j])*absolute_value(div1[i][j])
            E2[i] += absolute_value(div2[i][j])*absolute_value(div2[i][j])
            E3[i] += absolute_value(div3[i][j])*absolute_value(div3[i][j])
        E1[i] = -E1[i]/2
        E2[i] = -E2[i]/2
        E3[i] = -E3[i]/2
        P1[i] = np.exp(E1[i].data)
        P2[i] = np.exp(E2[i].data)
        P3[i] = np.exp(E3[i].data)
        if max(P1[i],P2[i],P3[i]) == P1[i]:
            P[i] = 1
        elif P2[i] - P3[i] > 0:
            P[i] = 2
        else:
            P[i] = 3
        print("P[i]: "+str(P[i]))