import numpy as np
# import quaternionic
import networkx as nx
from quaternion_calculation import *

"""
x0 = Variable(np.array([2.0,4,6,8]))
x1 = Variable(np.array([3,4,5,6]))
x2 = Variable(np.array(3))
x3 = Variable(np.array(5))
print(x2.size)
print(x0/x1)
print(x1/x0)
print(x2/x3)
print(x3/2)
print(-4/x2)
print(2/x0)

print('-------------')
x0 = Variable(np.array(3.0))
x1 = 4
x2 = Variable(np.array([1,2,3,4]))
x3 = Variable(np.array([5,6,7,8]))
print(x2*x2)
print(x3*x2)
print(rot(x2,x3))
print(conjugate(x2))

a = np.array([5,6,7,8])
b = Variable(a)
print(b)

c = np.array([[x2],[x3]])
d = np.array([[x2],[x3]])
print(np.dot(c,x2, c))

Win_num = np.random.uniform(-1, 1, (10, 1*4))
print(Win_num)
Win = np.full(10,Variable(np.array([0,0,0,0])))
for i in range(10):
    Win[i] = Variable(Win_num[i])
print(Win.size)
data = Variable(np.array([1,2,3,4]))

print(rot_arr(data, Win))
"""
# N_x = 5
# density = 0.2
# rho = 0.9
# m = int(N_x*(N_x-1)*density/2)  # 総結合数
# G = nx.gnm_random_graph(N_x, m, 0)

# # 行列への変換(結合構造のみ）
# connection = nx.to_numpy_matrix(G)
# W = np.array(connection)

# # 非ゼロ要素を一様分布に従う乱数として生成
# rec_scale = 1.0
# np.random.seed(seed=0)
# Win_num = np.random.uniform(-rec_scale, rec_scale, (N_x, N_x, 4))
# Win = np.full((N_x, N_x), Variable(np.array([0,0,0,0])))
# for i in range(N_x):
#     for j in range(N_x):
#         Win[i][j] = Variable(Win_num[i][j])
"""
for i in range(N_x):
    for j in range(N_x):
        Win[i][j] = W[i][j] * Win[i][j]

Win *= W
print(Win)
print("----------------------")
Win *= 0.1
"""
# a = Variable(np.array([1,2,3,4]))
# b = Variable(np.array([2,3,4,5]))
# c = Variable(np.array([3,4,5,6]))
# d = Variable(np.array([4,5,6,7]))
# e = np.array([[a,b],[c,d]])
# f = np.array([[b,c],[d,a]])
# g = np.array([[b],[d]])
# h = np.array([b,d])
# i = [a, b]
# j = [c, d]
# print(i[0])
# print(a*b+b*d)
# print(a*c+b*a)
# print(c*b+d*d)
# print(c*c+d*a)
# print("----------------------")
# print(np.dot(e, h))

# Wout_num = np.random.normal(size=(3,3,4))
# Wout = np.full((3, 3), Variable(np.array([0,0,0,0])))
# for i in range(3):
#     for j in range(3):
#         Wout[i][j] = Variable(Wout_num[i][j])
# print(Wout)

# hoge = np.full(10, Variable(np.array([1,2,3,4])))
# hoge_cut = hoge[:3]
# print(hoge_cut)
# poyo = np.array([np.array([1,2,3]),np.array([1,2,3])])
# print(poyo.shape[1])

# a = [5,6,7,8]
# a = np.array(a)
# b = Variable(a)
# print(b)

# def make_qvector(N_x, N_u, input_scale, seed=0):
#     # 一様分布に従う乱数
#     np.random.seed(seed=seed)
#     Win_num = np.random.uniform(-input_scale, input_scale, (N_x, N_u*4))
#     Win = np.full(N_x,Variable(np.array([0,0,0,0])))
#     for i in range(N_x):
#         Win[i] = Variable(Win_num[i])
#     return Win

# print(make_qvector(10, 1, 1))

# d = Variable(np.array([4,5,6,7]))
# print(d[0])

def to_quaternion(data):
    data_len = len(data)
    qdata = np.full(data_len, Variable(np.array([0,0,0,0])))
    for i in range(data_len):
        tmp = np.insert(data[i], 0, 0)
        print(tmp)
        qdata[i] = Variable(tmp)
    return qdata

# print(to_quaternion(np.array([[1,2,3],[5,6,7]])))
x0 = Variable(np.array([0,1,6,8]))
x1 = Variable(np.array([0,2,3.2,4.4]))
x2 = Variable(np.array([0,5,6,7]))
qdata = np.array([x0, x1, x2])

def to_3Ddata(qdata):
    data_len = len(qdata)
    data = np.full((data_len, 3), np.array([0, 0, 0]))
    for i in range(data_len):
        for j in range(3):
            data[i][j] = qdata[i].data[j+1]
    return data

print(qdata)
print(to_3Ddata(qdata))

N_x = 10
rho = 2

s = [1,2,3,4,5]
print(s[-2:])