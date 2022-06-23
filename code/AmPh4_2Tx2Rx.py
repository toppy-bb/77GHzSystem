import numpy as np
import matplotlib.pyplot as plt
import os

date = '20220623'
action = 'test'
round = '1'
#ai = 45
hp = 750
tmax = hp/25
vmax =300

#Vo =[2475,2467,2503,2492]#0112
Vo =[2474,2469,2503,2492]#0114-0119-0120

f = open(os.path.join(os.getcwd(),"data",'{0}_{1}_{2}.csv'.format(date, action, round)), 'r')
df = np.genfromtxt(os.path.join(os.getcwd(),"data",'{0}_{1}_{2}.csv'.format(date, action, round)), delimiter=",",
                   skip_header=1)  # np.genfromtxt()を使うと欠損値がnp.nanとして読み込まれる

fig = plt.figure(figsize=(12, 4))  # Figureを設定
plt.subplots_adjust(wspace=0.4, hspace=0.6)

t=df[:,0]/25
# Tx1-Rx1 Svv=I1+jQ1
I1 = df[:, 1] - Vo[0]
Q1 = df[:, 2] - Vo[1]
A1 = np.sqrt((I1 **2 + Q1 ** 2))
P1=(np.arctan2(Q1,I1))
# Tx1-Rx2 Shv=I2+jQ2
I2 = df[:, 3] - Vo[2]
Q2 = df[:, 4] - Vo[3]
A2 = np.sqrt((I2 ** 2 + Q2 ** 2))
P2=(np.arctan2(Q2,I2))
# Tx2-Rx1 Svh=I3+jQ3
I3 = -df[:, 5] -Vo[0]
Q3 = -df[:, 6] - Vo[1]
A3 = np.sqrt((I3 ** 2 + Q3 ** 2))
P3=(np.arctan2(Q3,I3))
# Tx2-Rx2 Shh=I4+jQ4
I4 = -df[:, 7] - Vo[2]
Q4 = -df[:, 8] - Vo[3]
A4 = np.sqrt((I4 ** 2 + Q4 ** 2))
P4=(np.arctan2(Q4,I4))


#Tx1-Rx1
ax1 = fig.add_subplot(1, 4, 1)   #1行4列の1番目
cm = plt.cm.get_cmap('hsv') # カラーマップ
mappable = ax1.scatter(t, A1, c=P1, vmin=-(np.pi), vmax=np.pi, s=3, cmap=cm)
#fig.colorbar(mappable, ax=ax1 ) # カラーバーを付加
plt.title("V-V",fontsize=12)
plt.xlabel("time [s]", fontsize=10)
plt.ylabel("amplitude [mV]", fontsize=10)
plt.xlim(0,tmax)
plt.ylim(0,vmax)

#Tx1-Rx2
ax2 = fig.add_subplot(1, 4, 2)   #1行4列の2番目
cm = plt.cm.get_cmap('hsv') # カラーマップ
mappable = ax2.scatter(t, A2, c=P2, vmin=-(np.pi), vmax=np.pi, s=3, cmap=cm)
fig.colorbar(mappable, ax=ax2 ) # カラーバーを付加
plt.title("H-V",fontsize=12)
plt.xlabel("time [s]", fontsize=10)
plt.ylabel("amplitude [mV]", fontsize=10)
plt.xlim(0,tmax)
plt.ylim(0,vmax)

#Tx2-Rx1
ax3 = fig.add_subplot(1, 4, 3)   #1行4列の3番目
cm = plt.cm.get_cmap('hsv') # カラーマップ
mappable = ax3.scatter(t, A3, c=P3, vmin=-(np.pi), vmax=np.pi, s=3, cmap=cm)
fig.colorbar(mappable, ax=ax2 ) # カラーバーを付加
plt.title("V-H",fontsize=12)
plt.xlabel("time [s]", fontsize=10)
plt.ylabel("amplitude [mV]", fontsize=10)
plt.xlim(0,tmax)
plt.ylim(0,vmax)

#Tx2-Rx2
ax4 = fig.add_subplot(1, 4, 4)   #1行4列の4番目
cm = plt.cm.get_cmap('hsv') # カラーマップ
mappable = ax4.scatter(t, A4, c=P4, vmin=-(np.pi), vmax=np.pi, s=3, cmap=cm)
fig.colorbar(mappable, ax=ax4 ) # カラーバーを付加
plt.title("H-H",fontsize=12)
plt.xlabel("time [s]", fontsize=10)
plt.ylabel("amplitude [mV]", fontsize=10)
plt.xlim(0,tmax)
plt.ylim(0,vmax)


plt.savefig(os.path.join(os.getcwd(),"img","{0}_{1}_{2}_AmPh4_2Tx2Rx".format(date, action, round)))
plt.show()