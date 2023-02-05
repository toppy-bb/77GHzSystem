import numpy as np
import matplotlib.pyplot as plt
import os
import sys

#date = '20221006'
#action = 'air'
#round = '2'
#freq = 1000

f = sys.argv[1]
freq =int(sys.argv[2])
#f = open(os.path.join(os.getcwd(),"data",'{0}_{1}_{2}.dat'.format(date, action, round)), 'r')
#df = np.genfromtxt(os.path.join(os.getcwd(),"data",'{0}_{1}_{2}.dat'.format(date, action, round)), delimiter=",")
df = np.genfromtxt(os.path.join(f), delimiter=",")
df = (df-2048)/2048*100 #DA変換

m_num = df.shape[0] #計測回数
tmax = m_num/freq #計測時間


fig = plt.figure(figsize=(13, 4),dpi=100)  # Figureを設定
plt.subplots_adjust(wspace=0.4, hspace=0.6)

if (np.sum(np.abs(df[0])) > np.sum(np.abs(df[1]))):
    # Tx1-Rx1 Svv=I1+jQ1
    I1 = df[0::2, 0]
    Q1 = df[0::2, 1]
    A1 = np.sqrt((I1 **2 + Q1 ** 2))
    P1=(np.arctan2(Q1,I1))
    # Tx1-Rx2 Svh=I2+jQ2
    I2 = df[0::2, 2]
    Q2 = df[0::2, 3]
    A2 = np.sqrt((I2 ** 2 + Q2 ** 2))
    P2=(np.arctan2(Q2,I2))
    # Tx2-Rx1 Shv=I3+jQ3
    I3 = df[1::2, 0]
    Q3 = df[1::2, 1]
    A3 = np.sqrt((I3 ** 2 + Q3 ** 2))
    P3=(np.arctan2(Q3,I3))
    # Tx2-Rx2 Shh=I4+jQ4
    I4 = df[1::2, 2]
    Q4 = df[1::2, 3]
    A4 = np.sqrt((I4 ** 2 + Q4 ** 2))
    P4=(np.arctan2(Q4,I4))

else:
        # Tx1-Rx1 Svv=I1+jQ1
    I1 = df[1::2, 0]
    Q1 = df[1::2, 1]
    A1 = np.sqrt((I1 **2 + Q1 ** 2))
    P1=(np.arctan2(Q1,I1))
    # Tx1-Rx2 Shv=I2+jQ2
    I2 = df[1::2, 2]
    Q2 = df[1::2, 3]
    A2 = np.sqrt((I2 ** 2 + Q2 ** 2))
    P2=(np.arctan2(Q2,I2))
    # Tx2-Rx1 Svh=I3+jQ3
    I3 = df[0::2, 0]
    Q3 = df[0::2, 1]
    A3 = np.sqrt((I3 ** 2 + Q3 ** 2))
    P3=(np.arctan2(Q3,I3))
    # Tx2-Rx2 Shh=I4+jQ4
    I4 = df[0::2, 2]
    Q4 = df[0::2, 3]
    A4 = np.sqrt((I4 ** 2 + Q4 ** 2))
    P4=(np.arctan2(Q4,I4))

vmax = max(A1.max(),A2.max(),A3.max(),A4.max())
vmin = max(A1.min(),A2.min(),A3.min(),A4.min())

t1= np.arange(0,np.size(A1))/freq*2
t2= np.arange(0,np.size(A3))/freq*2

print("計測時間: "+str(tmax)+"s")
print("サンプリングレート: "+str(freq)+"Hz")

#Tx1-Rx1
ax1 = fig.add_subplot(1, 4, 1)   #1行4列の1番目
cm = plt.cm.get_cmap('hsv') # カラーマップ
mappable = ax1.scatter(t1, A1, c=P1, vmin=-(np.pi), vmax=np.pi, s=3, cmap=cm)
#fig.colorbar(mappable, ax=ax1 ) # カラーバーを付加
plt.title("VV",fontsize=12)
plt.xlabel("time [s]", fontsize=10)
plt.ylabel("amplitude [mV]", fontsize=10)
plt.xlim(0,tmax)
plt.ylim(vmin-5,vmax+1)

#Tx1-Rx2
ax2 = fig.add_subplot(1, 4, 2)   #1行4列の2番目
cm = plt.cm.get_cmap('hsv') # カラーマップ
mappable = ax2.scatter(t1, A2, c=P2, vmin=-(np.pi), vmax=np.pi, s=3, cmap=cm)
fig.colorbar(mappable, ax=ax2 ) # カラーバーを付加
plt.title("VH",fontsize=12)
plt.xlabel("time [s]", fontsize=10)
plt.ylabel("amplitude [mV]", fontsize=10)
plt.xlim(0,tmax)
plt.ylim(vmin-5,vmax)

#Tx2-Rx1
ax3 = fig.add_subplot(1, 4, 3)   #1行4列の3番目
cm = plt.cm.get_cmap('hsv') # カラーマップ
mappable = ax3.scatter(t2, A3, c=P3, vmin=-(np.pi), vmax=np.pi, s=3, cmap=cm)
#fig.colorbar(mappable, ax=ax2 ) # カラーバーを付加
plt.title("HV",fontsize=12)
plt.xlabel("time [s]", fontsize=10)
plt.ylabel("amplitude [mV]", fontsize=10)
plt.xlim(0,tmax)
plt.ylim(vmin-5,vmax)

#Tx2-Rx2
ax4 = fig.add_subplot(1, 4, 4)   #1行4列の4番目
cm = plt.cm.get_cmap('hsv') # カラーマップ
mappable = ax4.scatter(t2, A4, c=P4, vmin=-(np.pi), vmax=np.pi, s=3, cmap=cm)
fig.colorbar(mappable, ax=ax4 ) # カラーバーを付加
plt.title("HH",fontsize=12)
plt.xlabel("time [s]", fontsize=10)
plt.ylabel("amplitude [mV]", fontsize=10)
plt.xlim(0,tmax)
plt.ylim(vmin-5,vmax)

savename = os.path.basename(f).strip('.dat')

plt.savefig(os.path.join("img", savename+"_AmPh_"+str(freq)+"Hz"))
plt.show()

fig1 = plt.figure(facecolor="white", figsize=(16, 12),dpi=100)
ax1 = fig1.add_subplot(projection="3d")
ax1.view_init(elev=30, azim=-130)
ax1.set_box_aspect((3,1,1))
ax1.set_title("$S_\mathrm{VV}$",size = 25)
ax1.set_xlabel("time [s]",size=15)
ax1.set_ylabel("Re [mV]",size=15)
ax1.set_zlabel("Im [mV]",size=15)
ax1.plot(t1,I1,Q1,lw=0.5)
#ax1.scatter(I1,Q1,lw=0.5)
plt.tight_layout()
plt.savefig(os.path.join("img", savename+"_Svv"))
plt.show()

fig2 = plt.figure(facecolor="white", figsize=(16, 12),dpi=100)
ax2 = fig2.add_subplot(projection="3d")
ax2.view_init(elev=30, azim=-130)
ax2.set_box_aspect((3,1,1))
ax2.set_title("$S_\mathrm{VH}$",size = 25)
ax2.set_xlabel("time [s]",size=15)
ax2.set_ylabel("Re [mV]",size=15)
ax2.set_zlabel("Im [mV]",size=15)
ax2.plot(t1,I2,Q2,lw=0.5)
#ax2.scatter(I2,Q2,lw=0.5)
plt.tight_layout()
plt.savefig(os.path.join("img", savename+"_Svh"))
plt.show()

fig3 = plt.figure(facecolor="white", figsize=(16, 12),dpi=100)
ax3 = fig3.add_subplot(projection="3d")
ax3.view_init(elev=30, azim=-130)
ax3.set_box_aspect((3,1,1))
ax3.set_title("$S_\mathrm{HV}$", size = 25)
ax3.set_xlabel("time [s]",size=15)
ax3.set_ylabel("Re [mV]",size=15)
ax3.set_zlabel("Im [mV]",size=15)
ax3.plot(t2,I3,Q3,lw=0.5)
#ax3.scatter(I3,Q3,lw=0.5)
plt.tight_layout()
plt.savefig(os.path.join("img", savename+"_Shv"))
plt.show()

fig4 = plt.figure(facecolor="white", figsize=(16, 12),dpi=100)
ax4 = fig4.add_subplot(projection="3d")
ax4.view_init(elev=30, azim=-130)
ax4.set_box_aspect((3,1,1))
ax4.set_title("$S_\mathrm{HH}$",size = 25)
ax4.set_xlabel("time [s]",size=15)
ax4.set_ylabel("Re [mV]",size=15)
ax4.set_zlabel("Im [mV]",size=15)
ax4.plot(t2,I4,Q4,lw=0.5)
#ax4.scatter(I4,Q4,lw=0.5)
plt.tight_layout()
plt.savefig(os.path.join("img", savename+"_Shh"))
plt.show()