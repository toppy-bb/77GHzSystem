import matplotlib.pyplot as plt
import numpy as np
import os
#from mpl_toolkits.mplot3d import Axes3D

date = '20221006'
action = 'air'
round = '2'
#ai = 45
#hp = 750
freq = 1000 # frequency [Hz]
#vmax =300

#Vo =[2475,2467,2503,2492]#0112
#Vo =[2474,2469,2503,2492,-2474,-2469,-2503,-2492]#0114-0119-0120

f = open(os.path.join(os.getcwd(),"data",'{0}_{1}_{2}.dat'.format(date, action, round)), 'r')
df = np.genfromtxt(os.path.join(os.getcwd(),"data",'{0}_{1}_{2}.dat'.format(date, action, round)), delimiter=",")  # np.genfromtxt()を使うと欠損値がnp.nanとして読み込まれる

df = (df-2048)/2048*2500


#DCshift
m_num = df.shape[0] #計測回数
tmax = m_num/freq #計測時間


fig = plt.figure(figsize=(16, 12))

#t=df[:,0]/10.7


# Tx1-Rx1 Svv=I1+jQ1
I1 = df[0::2, 0]
Q1 = df[0::2, 1]
# Tx1-Rx2 Shv=I2+jQ2
I2 = df[0::2, 2]
Q2 = df[0::2, 3]
# Tx2-Rx1 Svh=I3+jQ3
I3 = df[1::2, 0]
Q3 = df[1::2, 1]
# Tx2-Rx2 Shh=I4+jQ4
I4 = df[1::2, 2]
Q4 = df[1::2, 3]

t1= np.arange(0,np.size(I1))/freq*2
t2= np.arange(0,np.size(I3))/freq*2

ax1 = fig.add_subplot(projection="3d")
ax1.view_init(elev=30, azim=-135)
ax1.set_box_aspect((10,5,5))
ax1.set_title("Svv")
ax1.set_xlabel("time [s]")
ax1.set_ylabel("Re [mV]")
ax1.set_zlabel("Im [mV]")

# ax1.set_xlim(0,10)
# ax1.set_ylim(2400,2700)
ax1.plot(t1,I1,Q1)

""" ax2 = fig.add_subplot(222,projection="3d")
ax2.set_title("Shv")
ax2.set_xlabel("time [s]")
ax2.set_ylabel("Re [mV]")
ax2.set_zlabel("Im [mV]")
# ax2.set_xlim(0,10)
# ax2.set_ylim(2400,2700)
ax2.plot(t1[:200],I2[:200],Q2[:200])

ax3 = fig.add_subplot(223,projection="3d")
ax3.set_title("Svh")
ax3.set_xlabel("time [s]")
ax3.set_ylabel("Re [mV]")
ax3.set_zlabel("Im [mV]")
# ax3.set_xlim(0,10)
# ax3.set_ylim(2400,2700)
ax3.plot(t2[:200],I3[:200],Q3[:200])

ax4 = fig.add_subplot(224,projection="3d")
ax4.set_title("Shh")
ax4.set_xlabel("time [s]")
ax4.set_ylabel("Re [mV]")
ax4.set_zlabel("Im [mV]")
# ax4.set_xlim(0,10)
# ax4.set_ylim(2400,2700)
ax4.plot(t2[:200],I4[:200],Q4[:200]) """

plt.savefig(os.path.join(os.getcwd(),"img","{0}_{1}_{2}_IQ4_2Tx2Rx".format(date, action, round)))
plt.show()