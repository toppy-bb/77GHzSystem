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
# ファイル名は"s1_u1_d0.mat"のようになっており，
# sはspeaker (s1,s2,s5,s6,s7)
# uはmotion (u1, u2, ..., u9, u10)
# dはlabel (d0, d1, ..., d9)
# をそれぞれ表す．
# 
# 各.matファイルは構造体'spec'に行列データをもつ．
# 行列サイズは，チャネル数（n_channel=77）× 時間長（n_tau=データ依存）．
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
from scipy.io import loadmat
from rc_model import ESN
from sklearn.metrics import confusion_matrix


np.random.seed(seed=0)

# 音声信号を前処理したデータ(コクリアグラム)の読み込み
def read_motion_data(dir_name, motion_train_list):
    ''' 
    :入力：データファイル(.mat)の入っているディレクトリ名(dir_name)
    :出力：入力データ(input_data)と教師データ(teacher_data)
    '''
    # .matファイルのみを取得
    data_files = glob.glob(os.path.join(dir_name, '*.csv'))

    # データに関する情報
    n_time = 200  # サンプル数
    n_motion = 3  # ラベル数(motionの数)
    N_u = 24
    N_y = 3
    # q = 4 # quaternion

    # 初期化
    train_input = np.empty((0, N_u))  # 教師入力
    train_output = np.empty((0, N_y))  # 教師出力
    train_length = np.empty(0, np.int)  # データ長
    train_label = np.empty(0, np.int)  # 正解ラベル
    test_input = np.empty((0, N_u))  # 教師入力
    test_output = np.empty((0, N_y))  # 教師出力
    test_length = np.empty(0, np.int)  # データ長
    test_label = np.empty(0, np.int)  # 正解ラベル
    
    # データ読み込み
    if len(data_files) > 0:
        print("%d files in %s を読み込んでいます..." \
              % (len(data_files), dir_name))
        for each_file in data_files:
            # data = loadmat(each_file)
            df = pd.read_csv(os.path.join(each_file),header=None)
            data = np.array(df)
            # data = x.reshape(x.shape[0], x.shape[1]//4, 4)
            times = int(each_file[-5])  # 各データの動作番号
            # print(times)
            motion = {"standing":1, "walking":2, "sidewalking":3}
            label = motion[each_file[7:-7]] # クラスラベル
            if times in motion_train_list:  # 訓練用
                # 入力データ
                train_input = np.vstack((train_input, data))
                # 出力データ（n_tau x 10，値はすべて'-1'）
                tmp = -np.ones((n_time, N_y))
                tmp[:, label-1] = 1  # labelの列のみ1
                train_output = np.vstack((train_output, tmp))
                # データ長
                train_length = np.hstack((train_length, n_time))
                # 正解ラベル
                train_label = np.hstack((train_label, label))
            else:  # 検証用
                # 入力データ（構造体'spec'に格納されている）
                test_input = np.vstack((test_input, data))
                # 出力データ（n_tau x 10，値はすべて'-1'）
                tmp = -np.ones([n_time, N_y])
                tmp[:, label-1] = 1  # labelの列のみ1
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


N_x_list = np.arange(300, 400, 10)
for N_x in N_x_list:
    # 訓練データ，検証データの取得
    n_label = 3  # ラベル数
    train_list = [1, 2, 3, 4, 5]  # u1-u5が訓練用，残りが検証用
    train_input, train_output, train_length, train_label, test_input, \
    test_output, test_length, test_label = \
    read_motion_data(dir_name='./data',
                     motion_train_list=train_list)
    print("データ読み込み完了．訓練と検証を行っています...")


    # N_x = 100  # リザバーの大きさ
    print("リザバーの大きさ: %d" % N_x)
    train_WER = np.empty(0)
    test_WER = np.empty(0)

    # ESNモデル
    model = ESN(train_input.shape[1], train_output.shape[1], N_x,
                density=1, input_scale=1.0e+2, rho=0.1, fb_scale=0.0)

    ########## 訓練データに対して
    # リザバー状態行列
    stateCollectMat = np.empty((0, N_x))
    for i in range(len(train_input)):
        u_in = model.Input(train_input[i])
        r_out = model.Reservoir(u_in)
        stateCollectMat = np.vstack((stateCollectMat, r_out))
    
    # 教師出力データ行列
    teachCollectMat = train_output
    
    # 学習（疑似逆行列）
    Wout = np.dot(teachCollectMat.T, np.linalg.pinv(stateCollectMat.T))

    # ラベル出力
    Y_pred = np.dot(Wout, stateCollectMat.T)
    pred_train = np.empty(0, np.int)
    start = 0
    for i in range(len(train_length)):
        tmp = Y_pred[:,start:start+train_length[i]]  # 1つのデータに対する出力
        max_index = np.argmax(tmp, axis=0)+1  # 最大出力を与える出力ノード番号
        histogram = np.bincount(max_index)  # 出力ノード番号のヒストグラム
        pred_train = np.hstack((pred_train, np.argmax(histogram)))  # 最頻値 
        start = start + train_length[i]
    
    # 訓練誤差(Word Error Rate, WER)
    count = 0
    for i in range(len(train_length)):
        if pred_train[i] != train_label[i]:
            count = count + 1 
    print("訓練誤差： WER = %5.4lf" % (count/len(train_length)))
    train_WER = np.hstack((train_WER, count/len(train_length)))

    # 混同行列
    cm_train = confusion_matrix(train_label-1, pred_train-1, range(n_label))
    print("正解率： ACC = %5.4lf" % (np.sum(np.diag(cm_train))/len(test_length)))
    print(cm_train)

    ########## 検証データに対して
    # リザバー状態行列
    stateCollectMat = np.empty((0, N_x))
    for i in range(len(test_input)):
        u_in = model.Input(test_input[i])
        r_out = model.Reservoir(u_in)
        stateCollectMat = np.vstack((stateCollectMat, r_out))

    # ラベル出力
    Y_pred = np.dot(Wout, stateCollectMat.T)
    pred_test = np.empty(0, np.int)
    start = 0
    for i in range(len(test_length)):
        tmp = Y_pred[:,start:start+test_length[i]]  # 1つのデータに対する出力
        max_index = np.argmax(tmp, axis=0)+1  # 最大出力を与える出力ノード番号
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
    cm_test = confusion_matrix(test_label-1, pred_test-1, range(n_label))
    print("検証誤差： WER = %5.4lf" % (count/len(test_length)))
    print("正解率： ACC = %5.4lf" % (np.sum(np.diag(cm_test))/len(test_length)))
    print(cm_test)
    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(11, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.text(-0.2, 1.1, '(a)', transform=ax1.transAxes)
    ax1.text(0.4, 1.05, 'Training', transform=ax1.transAxes)
    df_cm_train = pd.DataFrame(cm_train, range(1,n_label+1), range(1,n_label+1))
    sn.heatmap(df_cm_train, cmap='Greys', annot=True, linewidths=1,
               linecolor='black', cbar=False, square=True, ax=ax1)
    plt.xlabel('Prediction')
    plt.ylabel('True')
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.text(-0.2, 1.1, '(b)', transform=ax2.transAxes)
    ax2.text(0.4, 1.05, 'Testing', transform=ax2.transAxes)
    df_cm_test = pd.DataFrame(cm_test, range(1,n_label+1), range(1,n_label+1))
    sn.heatmap(df_cm_test, cmap='Greys', annot=True, linewidths=1,
               linecolor='black', cbar=False, square=True, ax=ax2)
    plt.xlabel('Prediction')
    plt.ylabel('True')
    plt.savefig(os.path.join("img", "rc_Node"+str(N_x)))
    # plt.show()

