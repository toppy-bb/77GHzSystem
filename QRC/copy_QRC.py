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


np.random.seed(seed=0)

# Lorenz方程式によるデータ生成
class Lorenz:
    # パラメータの指定
    def __init__(self, sigma, r, b):
        self.sigma = sigma
        self.r = r
        self.b = b

    def f1(self, t, x, y, z):
        return -self.sigma*x + self.sigma*y

    def f2(self, t, x, y, z):
        return -x*z + self.r*x - y

    def f3(self, t, x, y, z):
        return x*y - self.b*z

    def Lorenz(self, t, X):
        '''
        :param t: 時間
        :param X: 3次元ベクトル
        :return: 3次元ベクトル
        '''
        next_X = [self.f1(t, X[0], X[1], X[2]), 
                  self.f2(t, X[0], X[1], X[2]), 
                  self.f3(t, X[0], X[1], X[2])]
        return np.array(next_X)

    # 4次のRunge-Kutta法により数値積分
    def Runge_Kutta(self, x0, T, dt):
        '''
        :param x0: 初期値
        :param T: 時間
        :param dt: ステップ幅
        :return: Lorenz方程式の時系列, (T/dt)x3
        '''
        X = x0
        t = 0
        data = []

        while t < T:
            k_1 = self.Lorenz(t, X)
            k_2 = self.Lorenz(t + dt/2, X + dt*k_1/2)
            k_3 = self.Lorenz(t + dt/2, X + dt*k_2/2)
            k_4 = self.Lorenz(t + dt, X + dt*k_3)
            next_X = X + dt/6*(k_1 + 2*k_2 + 2*k_3 + k_4)
            data.append(next_X)
            X = next_X
            t = t + dt

        return np.array(data)
    
    # ３次元データを四元数
    def to_quaternion(self, data):
        data_len = len(data)
        qdata = np.full(data_len, Variable(np.array([0,0,0,0])))
        for i in range(data_len):
            tmp = np.insert(data[i], 0, 0)
            qdata[i] = Variable(tmp)
        return qdata
    
    def to_3Ddata(self, qdata):
        data_len = len(qdata)
        data = np.full((data_len, 3), np.array([0, 0, 0]))
        for i in range(data_len):
            for j in range(3):
                data[i][j] = qdata[i].data[j+1]
        return data

if __name__ == '__main__':

    # Lorenz方程式から時系列データ生成
    T_train = 100  # 学習データの長さ
    T_test = 25  # テストデータの長さ
    dt = 0.02  # ステップ幅
    x0 = np.array([1, 0, 1])  # 初期値
    
    dynamics = Lorenz(10.0, 28.0, 8.0/3.0)
    data = dynamics.Runge_Kutta(x0, T_train + T_test, dt)
    qdata = dynamics.to_quaternion(data)

    # 訓練・検証用情報
    train_U = data[:int(T_train/dt)]
    train_QU = qdata[:int(T_train/dt)]
    train_D = data[1:int(T_train/dt)+1]
    train_QD = qdata[1:int(T_train/dt)+1]

    test_U = data[int(T_train/dt):int((T_train+T_test)/dt)]
    test_QU = qdata[int(T_train/dt):int((T_train+T_test)/dt)]
    test_D = data[1+int(T_train/dt):int((T_train+T_test)/dt)+1]
    test_QD = qdata[1+int(T_train/dt):int((T_train+T_test)/dt)+1]


    # ESNモデル
    N_x = 100  # リザバーのノード数
    rho = 0.01
    model = ESN(1, 1, N_x, \
                density=0.1, input_scale=0.1, rho=rho)

    # オンライン学習と予測
    # print(train_QU)
    # print(test_QU)
    qtrain_Y = model.qadapt(train_QU, train_QD, QGC(N_x, 0.01))
    # print(qtrain_Y)
    train_Y = dynamics.to_3Ddata(qtrain_Y)
    # print(train_Y)
    qtest_Y = model.qfree_run(test_QU)
    test_Y = dynamics.to_3Ddata(qtest_Y)

    print(x0)
    print("Node", N_x, "train_len", T_train)
    train_error = 0
    for i in range(len(train_Y)-750, len(train_Y)):
        for j in range(train_Y.shape[1]):
            train_error += (train_Y[i][j] - train_D[i][j])**2
            
    print("train_error", (train_error/750)**0.5)

    test_error = 0
    for i in range(len(test_Y)-1):
        for j in range(test_Y.shape[1]):
            test_error += (test_Y[i][j] - test_D[i][j])**2
    
    print("test_error", (test_error/(len(test_Y)-1))**0.5)
    
    """
    # 予測性能評価（valid time）
    eps = 1  # 許容誤差
    for n in range(int(T_test/dt)):
        dif = np.sqrt(((test_D[n,:] - test_Y[n,:]) ** 2).mean())
        if dif > eps:
            valid_time = float(n)*dt
            break

    print('valid time = ', valid_time)
    """
    # グラフ表示用データ
    T_disp = (-15, 15)
    t_axis = np.arange(T_disp[0], T_disp[1], dt)  # 時間軸
    disp_D = np.concatenate((train_D[int(T_disp[0]/dt):], 
                            test_D[:int(T_disp[1]/dt)]))  # 目標出力
    disp_Y = np.concatenate((train_Y[int(T_disp[0]/dt):], 
                            test_Y[:int(T_disp[1]/dt)]))  # モデル出力

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 7))
    plt.subplots_adjust(hspace=0.3)

    # ax1 = fig.add_subplot(3, 1, 1)
    # ax1.text(-0.15, 1, '(a)', transform=ax1.transAxes)
    # ax1.text(0.2, 1.05, 'Training', transform=ax1.transAxes)
    # ax1.text(0.7, 1.05, 'Testing', transform=ax1.transAxes)
    # plt.plot(t_axis, disp_D[:,0], color='k', label='Target')
    # plt.plot(t_axis, disp_Y[:,0], color='gray', linestyle='--', label='Model')
    # plt.ylabel('x')
    # plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')
    # plt.legend(bbox_to_anchor=(0, 0), loc='lower left')

    # ax2 = fig.add_subplot(3, 1, 2)
    # ax2.text(-0.15, 1, '(b)', transform=ax2.transAxes)
    # plt.plot(t_axis, disp_D[:,1], color='k', label='Target')
    # plt.plot(t_axis, disp_Y[:,1], color='gray', linestyle='--', label='Model')
    # plt.ylabel('y')
    # plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')
    # plt.legend(bbox_to_anchor=(0, 0), loc='lower left')

    # ax3 = fig.add_subplot(3, 1, 3)
    # ax3.text(-0.15, 1, '(c)', transform=ax3.transAxes)
    # plt.plot(t_axis, disp_D[:,2], color='k', label='Target')
    # plt.plot(t_axis, disp_Y[:,2], color='gray', linestyle='--', label='Model')
    # plt.ylabel('z')
    # plt.xlabel('t')
    # plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')
    # plt.legend(bbox_to_anchor=(0, 0), loc='lower left')

    # #plt.savefig(f'out/result_tanh_{N_x}node_sp{rho}.png')
    # plt.savefig(f'out/result.png')

    ax1 = fig.add_subplot(3, 1, 1)
    # ax1.text(-0.15, 1, '(a)', transform=ax1.transAxes)
    # ax1.text(0.2, 1.05, 'Training', transform=ax1.transAxes)
    # ax1.text(0.7, 1.05, 'Testing', transform=ax1.transAxes)
    ax1.set_ylim(-25, 25)
    ax1.set_yticks([-20, 0, 20])
    plt.plot(t_axis, disp_D[:,0], color='k', label='Target')
    plt.plot(t_axis, disp_Y[:,0], color='gray', linestyle='--', label='Model')
    # plt.ylabel('x')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    
    ax2 = fig.add_subplot(3, 1, 2)
    # ax2.text(-0.15, 1, '(b)', transform=ax2.transAxes)
    ax2.set_ylim(-25, 25)
    ax2.set_yticks([-20, 0, 20])
    plt.plot(t_axis, disp_D[:,1], color='k', label='Target')
    plt.plot(t_axis, disp_Y[:,1], color='gray', linestyle='--', label='Model')
    # plt.ylabel('y')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')

    ax3 = fig.add_subplot(3, 1, 3)
    # ax3.text(-0.15, 1, '(c)', transform=ax3.transAxes)
    ax3.set_ylim(-3, 50)
    ax3.set_yticks([0, 20, 40])
    plt.plot(t_axis, disp_D[:,2], color='k', label='Target')
    plt.plot(t_axis, disp_Y[:,2], color='gray', linestyle='--', label='Model')
    # plt.ylabel('z')
    plt.xlabel('Time t', fontsize=20)
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')

    # plt.savefig(f'out/result_Node{N_x}.png')
    plt.savefig(f'out/result_copy.png')

    # fig2 = plt.figure()
    # ax = fig2.add_subplot(111, projection='3d')
    # ax.set_xlabel("x", size = 14)
    # ax.set_ylabel("y", size = 14)
    # ax.set_zlabel("z", size = 14)

    # # 軸目盛を設定
    # # ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    # # ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    # ax.plot(disp_D[:,0], disp_D[:,1], disp_D[:,2], color="red", linewidth=0.5, label='Target')
    # ax.plot(disp_Y[:,0], disp_Y[:,1], disp_Y[:,2], color="blue", linewidth=0.5, label='Model')
    # plt.savefig(f'3d/qrc{N_x}.png')
