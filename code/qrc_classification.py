# !/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# spoken_confusion_matrix.py: 本書の図5.2に対応するサンプルコード
#################################################################

#################################################################
# ■ 音声データの準備
#
# https://github.com/dsiufl/Reservoir-Computing
# より，"Lyon_decimation_128"というフォルダをダウンロードして
# 同じフォルダに置く．
#
# これは，TI 46-Word Corpus
# https://catalog.ldc.upenn.edu/LDC93S9
# に含まれる音声データの一部をLyon's auditory modelで前処理した後の
# コクリアグラムのデータセットである（計500ファイル）．
# （注：オリジナルデータの使用にはLinguistic Data Consortiumのライセンスが必要）
#
# ファイル名は"{motion}_{time}.dat"のようになっており，
# motion =  (standing,walking,sidewalking)
# times = (01, 02, ..., 10)
# をそれぞれ表す．
# 
# 各.datファイルは
# data_file = [[input_quaternion (N_u*4), [output_quaternion (N_y*4)], label] * n_time]
# 行列サイズは，チャネル数（n_time=200）× 時間長（n_tau=データ依存）．
# 
# L115のtrain_listに含まれるmotionを訓練用，それ以外を検証用とする．
#
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sn
import pandas as pd
# from scipy.io import loadmat    
from qrc_model import ESN, SGD
from sklearn.metrics import confusion_matrix


np.random.seed(seed=0)

# 音声信号を前処理したデータ(コクリアグラム)の読み込み
def read_motion_data(dir_name, motion_train_list):
    ''' 
    :入力：データファイル(.dat)の入っているディレクトリ名(dir_name)
    :出力：入力データ(input_data)と教師データ(teacher_data)
    '''
    # .datファイルのみを取得
    data_files = glob.glob(os.path.join(dir_name, '*.csv'))

    # データに関する情報
    n_time = 200  # サンプル数
    n_motion = 3  # ラベル数(motionの数)
    N_u = 6
    N_y = 1
    q = 4 # quaternion

    # 初期化
    train_input = np.empty((0, N_u, q))  # 教師入力
    train_output = np.empty((0, N_y, q))  # 教師出力
    train_length = np.empty(0, np.int)  # データ長
    train_label = np.empty(0, np.int)  # 正解ラベル
    test_input = np.empty((0, N_u, q))  # 教師入力
    test_output = np.empty((0, N_y, q))  # 教師出力
    test_length = np.empty(0, np.int)  # データ長
    test_label = np.empty(0, np.int)  # 正解ラベル
    
    # データ読み込み
    if len(data_files) > 0:
        print("%d files in %s を読み込んでいます..." \
              % (len(data_files), dir_name))
        for each_file in data_files:
            # data = loadmat(each_file)
            df = pd.read_csv(os.path.join(each_file),header=None)
            x = np.array(df)
            data = x.reshape(x.shape[0], x.shape[1]//4, 4)
            times = int(each_file[-5])  # 各データの動作番号
            # print(times)
            motion = {"standing":1, "walking":2, "sidewalking":3}
            label = motion[each_file[7:-7]] # クラスラベル
            if times in motion_train_list:  # 訓練用
                # 入力データ
                train_input = np.vstack((train_input, data)) # [N_u*4] * n_time
                # 出力データ（n_time x n_label，値はすべて'-1'）
                tmp = -np.ones((n_time, 1, 4))
                tmp[:, :, label] = 1  # labelの列のみ1
                train_output = np.vstack((train_output, tmp))
                # データ長
                train_length = np.hstack((train_length, n_time))
                # 正解ラベル
                train_label = np.hstack((train_label, label))
            else:  # 検証用
                # 入力データ
                test_input = np.vstack((test_input, data)) # [N_u*4] * n_time
                # 出力データ（n_time x n_label，値はすべて'-1'）
                tmp = np.ones((n_time, 1, 4))
                tmp[:, :, label] = 1  # labelの列のみ1
                test_output = np.vstack((test_output, tmp))
                # データ長
                test_length = np.hstack((test_length, n_time))
                # 正解ラベル
                test_label = np.hstack((test_label, label))
    else:
        print("ディレクトリ %s にファイルが見つかりません．" % (dir_name))
        return
    return train_input, train_output, train_length, train_label, \
           test_input, test_output, test_length, test_label

def outer_product(a, b):
    res = [0]*4
    res[0] = a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]
    res[1] = a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2]
    res[2] = a[0]*b[2]+a[2]*b[0]+a[3]*b[1]-a[1]*b[3]
    res[3] = a[0]*b[3]+a[3]*b[0]+a[1]*b[2]-a[2]*b[1]
    return res

def inner_product_array(a,b):
    return np.sum(a*b,axis=1)

def inner_product(a,b):
    return np.sum(a*b)

def outer_product_array(a,b):
    return list(map(outer_product, a, b))

def norm_array(a):
    return np.linalg.norm(a,axis=1).reshape(a.shape[0],1)

def conjugation_array(a):
    for i in range(len(a)):
        a[i,1]*= -1
        a[i,2]*= -1
        a[i,3]*= -1
    return a

def conjugation(a):
    a[1]*= -1
    a[2]*= -1
    a[3]*= -1
    return a

def rot(W,x):
    s = np.empty((0,4))
    for i in range(len(W)):
        tmp = outer_product_array(outer_product_array(W[i],x),conjugation_array(W[i]))/norm_array(W[i])
        s = np.vstack((s, np.sum(tmp,axis=0)))
    return s

N_x_list = [10,15,20,25,30]
for N_x in N_x_list:
    # 訓練データ，検証データの取得
    n_motion = 3  # ラベル数
    train_list = [1, 2, 3, 4, 5]  # 01-05が訓練用，残りが検証用
    train_input, train_output, train_length, train_label, test_input, \
    test_output, test_length, test_label = \
    read_motion_data(dir_name='./data/',
                     motion_train_list=train_list)
    
    # print( train_label, test_label)
    print("データ読み込み完了．訓練と検証を行っています...")

    # N_x = 10 # リザバーの大きさ
    print("リザバーの大きさ: %d" % N_x)
    train_WER = np.empty(0)
    test_WER = np.empty(0)

    # ESNモデル
    model = ESN(train_input.shape[1], train_output.shape[1], N_x,
                density=1, input_scale=0.1, rho=0.8, fb_scale=0.0)

    ########## 訓練データに対して
    # リザバー状態行列
    # stateCollectMat = np.empty((0, N_x, 4))
    # for i in range(len(train_input)):
    #     u_in = model.Input(train_input[i])
    #     r_out = model.Reservoir(u_in)
    #     stateCollectMat = np.vstack((stateCollectMat, r_out))
    
    # 教師出力データ行列
    # teachCollectMat = train_output
    
    # 学習（疑似逆行列）
    # Wout = np.dot(teachCollectMat.T, np.linalg.pinv(stateCollectMat.T))

    

    # ラベル出力
    Y_pred, Wout = model.adapt(train_input,train_output,SGD(N_x, train_output.shape[1], lr=0.01))
    pred_train = np.empty(0, np.int)
    start = 0
    # print(len(train_length))
    for i in range(len(train_length)): # 訓練データの数
        tmp = Y_pred[start:start+train_length[i],0]  # 1つのデータに対する出力　??
        max_index = np.argmax(tmp[:,1:], axis=1)+1  # 最大出力を与える出力ノード番号
        histogram = np.bincount(max_index)  # 出力ノード番号のヒストグラム
        pred_train = np.hstack((pred_train, np.argmax(histogram)))  # 最頻値 
        start = start + train_length[i]
    # print(pred_train)
    # 訓練誤差(Word Error Rate, WER)
    count = 0
    for i in range(len(train_length)):
        if pred_train[i] != train_label[i]:
            count = count + 1 
    print("訓練誤差： WER = %5.4lf" % (count/len(train_length)))
    train_WER = np.hstack((train_WER, count/len(train_length)))
    # 混同行列
    cm_train = confusion_matrix(train_label-1, pred_train-1, range(n_motion))
    print("正解率： ACC = %5.4lf" % (np.sum(np.diag(cm_train))/len(test_length)))
    print(cm_train)
        
    ########## 検証データに対して
    # リザバー状態行列
    # stateCollectMat = np.empty((0, N_x, 4))
    for i in range(len(test_input)):
        u_in = model.Input(test_input[i])
        r_out = model.Reservoir(u_in)
        Y_pred[i] = rot(Wout, r_out)

    pred_test = np.empty(0, np.int)
    start = 0
    for i in range(len(test_length)):
        tmp = Y_pred[start:start+test_length[i],0]  # 1つのデータに対する出力
        max_index = np.argmax(tmp[:,1:], axis=1)+1  # 最大出力を与える出力ノード番号
        histogram = np.bincount(max_index)  # 出力ノード番号のヒストグラム
        pred_test = np.hstack((pred_test, np.argmax(histogram)))  # 最頻値
        start = start + test_length[i]
        
    # 検証誤差(WER)
    count = 0
    for i in range(len(test_length)):
        if pred_test[i] != test_label[i]:
            count = count + 1 
    
    test_WER = np.hstack((test_WER, count/len(test_length)))
    # 混同行列
    cm_test = confusion_matrix(test_label-1, pred_test-1, range(n_motion))
    print("検証誤差： WER = %5.4lf" % (count/len(test_length)))
    print("正解率： ACC = %5.4lf" % (np.sum(np.diag(cm_test))/len(test_length)))
    print(cm_test)
        
    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(11, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.text(-0.2, 1.1, '(a)', transform=ax1.transAxes)
    ax1.text(0.4, 1.05, 'Training', transform=ax1.transAxes)
    df_cm_train = pd.DataFrame(cm_train, range(1,n_motion+1), range(1,n_motion+1))
    sn.heatmap(df_cm_train, cmap='Greys', annot=True, linewidths=1,
               linecolor='black', cbar=False, square=True, ax=ax1)
    plt.xlabel('Prediction')
    plt.ylabel('True')
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.text(-0.2, 1.1, '(b)', transform=ax2.transAxes)
    ax2.text(0.4, 1.05, 'Testing', transform=ax2.transAxes)
    df_cm_test = pd.DataFrame(cm_test, range(1,n_motion+1), range(1,n_motion+1))
    sn.heatmap(df_cm_test, cmap='Greys', annot=True, linewidths=1,
               linecolor='black', cbar=False, square=True, ax=ax2)
    plt.xlabel('Prediction')
    plt.ylabel('True')

    plt.savefig(os.path.join("img", "qrc_Node"+str(N_x)))   
    plt.show()