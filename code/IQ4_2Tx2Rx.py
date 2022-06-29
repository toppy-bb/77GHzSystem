import matplotlib.pyplot as plt
import numpy as np
import os

date = '20220623'
action = 'test'
round = '1'

#Vo =[2475,2467,2503,2492]#0112
#Vo =[2474,2469,2503,2492]#0114

f = open(os.path.join(os.getcwd(),"data",'{0}_{1}_{2}.csv'.format(date, action, round)), 'r')
df = np.genfromtxt(os.path.join(os.getcwd(),"data",'{0}_{1}_{2}.csv'.format(date, action, round)), delimiter=",",
                   skip_header=1)  # np.genfromtxt()を使うと欠損値がnp.nanとして読み込まれる

print(type(df))
Vo = df.mean(axis = 0)[1:9]
print(Vo)

fig = plt.figure(figsize=(14, 8))

t=df[:,0]/10.7
# Tx1-Rx1 Svv=I1+jQ1
I1 = df[:, 1] - Vo[0]
Q1 = df[:, 2] - Vo[1]
# Tx1-Rx2 Shv=I2+jQ2
I2 = df[:, 3] - Vo[2]
Q2 = df[:, 4] - Vo[3]
# Tx2-Rx1 Svh=I3+jQ3
I3 = -df[:, 5] -Vo[4]
Q3 = -df[:, 6] - Vo[5]
# Tx2-Rx2 Shh=I4+jQ4
I4 = -df[:, 7] - Vo[6]
Q4 = -df[:, 8] - Vo[7]

ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title("Svv")
ax1.set_xlabel("Re")
ax1.set_ylabel("Im")
# ax1.set_xlim(0,10)
# ax1.set_ylim(2400,2700)
ax1.scatter(I1,Q1,s=20)

ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title("Shv")
ax2.set_xlabel("Re")
ax2.set_ylabel("Im")
# ax2.set_xlim(0,10)
# ax2.set_ylim(2400,2700)
ax2.scatter(I2,Q2,s=20)

ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title("Svh")
ax3.set_xlabel("Re")
ax3.set_ylabel("Im")
# ax3.set_xlim(0,10)
# ax3.set_ylim(2400,2700)
ax3.scatter(I3,Q3,s=20)

ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title("Shh")
ax4.set_xlabel("Re")
ax4.set_ylabel("Im")
# ax4.set_xlim(0,10)
# ax4.set_ylim(2400,2700)
ax4.scatter(I4,Q4,s=20)

plt.savefig(os.path.join(os.getcwd(),"img","{0}_{1}_{2}_IQ4_2Tx2Rx".format(date, action, round)))
plt.show()