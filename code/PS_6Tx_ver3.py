#色なし
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

date = '0125'
action = 'base'
round = '1'


Vo =[2475,2469,2503,2492]

f = open('{0}_{1}_{2}.csv'.format(date, action, round), 'r')
df = np.genfromtxt("{0}_{1}_{2}.csv".format(date, action, round), delimiter=",",
                   skip_header=1)  # np.genfromtxt()を使うと欠損値がnp.nanとして読み込まれる

# Tx1-Rx1 Svv=I1+jQ1
I1 = df[:, 1] - Vo[0]
Q1 = df[:, 2] - Vo[1]
A1 = np.sqrt((I1 **2 + Q1 ** 2))
# Tx1-Rx2 Shv=I2+jQ2
I2 = df[:, 3] - Vo[2]
Q2 = df[:, 4] - Vo[3]
A2 = np.sqrt((I2 ** 2 + Q2 ** 2))
# Tx2-Rx1 Svh=I3+jQ3
I3 = -df[:, 5] -Vo[0]
Q3 = -df[:, 6] - Vo[1]
A3 = np.sqrt((I3 ** 2 + Q3 ** 2))
# Tx2-Rx2 Shh=I4+jQ4
I4 = -df[:, 7] - Vo[2]
Q4 = -df[:, 8] - Vo[3]
A4 = np.sqrt((I4 ** 2 + Q4 ** 2))

#平均ストークスベクトルg

#送信-水平偏波
g0 = (A4**2)+(A3**2)
g1 = (A4**2)-(A3**2)
g2= 2*(I3*I4+Q3*Q4)
g3= 2*(I4*Q3-I3*Q4)
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


#図の作製
fig = plt.figure(figsize=(5,4))  # Figureを設定
plt.subplots_adjust(wspace=0.2, hspace=0.3)

#水平
ax1 = fig.add_subplot(1, 1, 1,projection='3d')   #2行3列の1番目
ax1.w_xaxis.set_pane_color((0., 0., 0., 0.))
ax1.w_yaxis.set_pane_color((0., 0., 0., 0.))
ax1.w_zaxis.set_pane_color((0., 0., 0., 0.))
ax1.set_title("{0}_{1}_{2}".format(date,action,round),fontsize=12)
ax1.set_xlabel("g1/g0")
ax1.set_ylabel("g2/g0")
ax1.set_zlabel("g3/g0")
ax1.grid(color="white")
ax1.grid(False)
# sphere
u,v=np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
x=np.cos(u)*np.sin(v)
y=np.sin(u)*np.sin(v)
z=np.cos(v)
ax1.plot_wireframe(x, y, z, color='0.8', linewidth=0.5,rstride =2, cstride = 2) # alpha=1
ax1.set_xticks(np.linspace(-1.0, 1.0, 5))
ax1.set_yticks(np.linspace(-1.0, 1.0, 5))
ax1.set_zticks(np.linspace(-1.0, 1.0, 5))

# plotdata
ax1.scatter(g1/g0, g2/g0, g3/g0,  vmin=-(np.pi), vmax=np.pi, s=8, c='red')
ax1.scatter(h1/h0, h2/h0, h3/h0,  vmin=-(np.pi), vmax=np.pi, s=8, c='blue')
ax1.scatter(m1/m0, m2/m0, m3/m0,  vmin=-(np.pi), vmax=np.pi, s=8, c='green')
ax1.scatter(n1/n0, n2/n0, n3/n0,  vmin=-(np.pi), vmax=np.pi, s=8, c='orange')
ax1.scatter(l1/l0, l2/l0, l3/l0,  vmin=-(np.pi), vmax=np.pi, s=8, c='black')
ax1.scatter(r1/r0, r2/r0, r3/r0,  vmin=-(np.pi), vmax=np.pi, s=8, c='purple')

plt.savefig("{0}_{1}_{2}_PS2".format(date,action,round))

print(sum(g0)/len(g0),sum(h0)/len(h0),sum(m0)/len(m0),sum(n0)/len(n0),sum(l0)/len(l0),sum(r0)/len(r0))

plt.show()
print("finished")

