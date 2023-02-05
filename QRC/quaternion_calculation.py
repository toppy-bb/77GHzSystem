import numpy as np
import copy

class TypesError(Exception):
    pass

# スカラーまたは四元数（nd.array型）
class Variable:
    __array_priority__ = 200
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.name = name
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.ndim
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    def __mul__(self, other):
        if self.size == 4 and (np.isscalar(other) or other.size == 1):
            # x : 四元数、s : 実数
            return scalar_multiply(x=self, s=other)
        elif (isinstance(other, Variable) and self.size == 1) and other.size == 4:
            # x : 実数、s : 四元数
            return scalar_multiply(x=other, s=self)
        elif self.size == 4 and other.size == 4:
            # x : 四元数、s :  四元数
            return outer_product(self, other)
        else:
            # x : 実数、s :  実数
            return real_multiply(self, other)
        
    def __rmul__(self, other):
        if self.size == 4 and (np.isscalar(other) or other.size == 1):
            # x : 四元数、s : 実数
            return scalar_multiply(x=self, s=other)
        elif (isinstance(other, Variable) and self.size == 1) and other.size == 4:
            # x : 実数、s : 四元数
            return scalar_multiply(x=other, s=self)
        elif self.size == 4 and other.size == 4:
            # x : 四元数、s :  四元数
            return outer_product(self, other)
        else:
            # x : 実数、s :  実数
            return real_multiply(self, other)
        
    
    def __add__(self, other):
        if self.size == 4 and other.size == 4:
            # x : 四元数、s :  四元数
            return quaternion_add(self, other)
        elif self.size == 1 and (np.isscalar(self) or self.size == 1):
            # x : 実数、s :  実数
            return real_add(self, other)
        else:
            raise TypesError()
        
    def __radd__(self, other):
        if self.size == 4 and other.size == 4:
            # x : 四元数、s :  四元数
            return quaternion_add(self, other)
        elif self.size == 1 and (np.isscalar(self) or self.size == 1):
            # x : 実数、s :  実数
            return real_add(self, other)
        else:
            raise TypesError()
        
    def __neg__(self, other):
        if np.isscalar(other) or self.size == 1:
            return real_neg(self)
        else:
            return quaternion_neg(self)
        
    def __sub__(self, other):
        if self.size == 4 and other.size == 4:
            # x : 四元数、s :  四元数
            return quaternion_subtraction(self, other)
        elif (np.isscalar(self) or self.size == 1) and (np.isscalar(other) or other.size == 1):
            # x : 実数、s :  実数
            return real_subtraction(self, other)
        else:
            raise TypesError()
        
    def __rsub__(self, other):
        if self.size == 4 and other.size == 4:
            # x : 四元数、s :  四元数
            return quaternion_rsubtraction(self, other)
        elif (np.isscalar(self) or self.size == 1) and (np.isscalar(other) or other.size == 1):
            # x : 実数、s :  実数
            return real_rsubtraction(self, other)
        else:
            raise TypesError()
    
    def __truediv__(self, other):
        if self.size == 4 and isinstance(other, Variable) and other.size == 4:
            return quaternion_div(self, other)
        elif self.size == 4 and ((isinstance(other, Variable) and other.size == 1) or np.isscalar(other)):
            return scalar_div(self, other)
        elif self.size == 1 and (isinstance(other, Variable) and other.size == 4):
            return quaternion_inv(self, other)
        else:
            return real_div(self, other)
    
    def __rtruediv__(self, other):
        if self.size == 4 and isinstance(other, Variable) and other.size == 4:
            return quaternion_rdiv(self, other)
        elif (self.size == 1 or np.isscalar(self)) and ((isinstance(other, Variable) and other.size == 4)):
            return scalar_rdiv(self, other)
        elif self.size == 4 and np.isscalar(other):
            return quaternion_rinv(self, other)
        else:
            return real_rdiv(self, other)
                
class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()

# 四元数マイナス
class QuaterionNeg(Function):
    def forward(self, x):
        output = np.array([0, 0, 0, 0], np.single)
        output[0] = -x[0]
        output[1] = -x[1]
        output[2] = -x[2]
        output[3] = -x[3]
        return output
    
# 実数マイナス
class RealNeg(Function):
    def forward(self, x):
        return -x

# 四元数演算
# スカラー倍
class ScalarMultiply(Function):
    def forward(self, x, s):
        output = np.array([0, 0, 0, 0], np.single)
        output[0] = x[0] * s
        output[1] = x[1] * s
        output[2] = x[2] * s
        output[3] = x[3] * s
        return output
    
# スカラー割り算   
class ScalarDiv(Function):
    def forward(self, x, s):
        output = np.array([0, 0, 0, 0], np.single)
        output[0] = x[0] / s
        output[1] = x[1] / s
        output[2] = x[2] / s
        output[3] = x[3] / s
        return output
    
# 複素共役
class Conjugate(Function):
    def forward(self, x):
        output = np.array([0, 0, 0, 0], np.single)
        output[0] = x[0]
        output[1] = -x[1]
        output[2] = -x[2]
        output[3] = -x[3]
        return output

# 絶対値
class AbsoluteValue(Function):
    def forward(self, x):
        output = (x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2)**0.5
        return output

# 四元数加算
class QuaternionAdd(Function):
    def forward(self, x0, x1):
        output = np.array([0, 0, 0, 0], np.single)
        for i in range(4):
            output[i] = x0[i] + x1[i]
        return output

# 実数加算
class RealAdd(Function):
    def forward(self, x0, x1):
        return x0 + x1
    
# 四元数引き算
class QuaternionSubtraction(Function):
    def forward(self, x0, x1):
        output = np.array([0, 0, 0, 0], np.single)
        for i in range(4):
            output[i] = x0[i] - x1[i]
        return output

# 実数引き算
class RealSubtraction(Function):
    def forward(self, x0, x1):
        return x0 - x1

# 外積
class OuterProduct(Function):
    def forward(self, x0, x1):
        output = np.array([0, 0, 0, 0], np.single)
        output[0] = x0[0]*x1[0] - x0[1]*x1[1] - x0[2]*x1[2] - x0[3]*x1[3]
        output[1] = x0[0]*x1[1] + x0[1]*x1[0] + x0[2]*x1[3] - x0[3]*x1[2]
        output[2] = x0[0]*x1[2] + x0[2]*x1[0] + x0[3]*x1[1] - x0[1]*x1[3]
        output[3] = x0[0]*x1[3] + x0[3]*x1[0] + x0[1]*x1[2] - x0[2]*x1[1]
        return output
    
# 内積
class InnerProduct(Function):
    def forward(self, x0, x1):
        output = x0[0]*x1[0] + x0[1]*x1[1] + x0[2]*x1[2] + x0[3]*x1[3]
        return output
    
# アダマール積
class HadamardProduct(Function):
    def forward(self, x0, x1):
        output = np.array([0, 0, 0, 0], np.single)
        for i in range(4):
            output[i] = x0[i]*x1[i]
        return output

# 実数積
class RealMultiply(Function):
    def forward(self, x0, x1):
        return x0 * x1
   
# 四元数割り算
class QuaternionDiv(Function):
    def forward(self, x0, x1):
        output = np.array([0, 0, 0, 0], np.single)
        x1_abs2 = x1[0]**2 + x1[1]**2 + x1[2]**2 + x1[3]**2
        output[0] = (x0[0]*x1[0] + x0[1]*x1[1] + x0[2]*x1[2] + x0[3]*x1[3]) / x1_abs2
        output[1] = (-x0[0]*x1[1] + x0[1]*x1[0] - x0[2]*x1[3] + x0[3]*x1[2]) / x1_abs2
        output[2] = (-x0[0]*x1[2] + x0[2]*x1[0] - x0[3]*x1[1] + x0[1]*x1[3]) / x1_abs2
        output[3] = (-x0[0]*x1[3] + x0[3]*x1[0] - x0[1]*x1[2] + x0[2]*x1[1]) / x1_abs2
        return output

# 四元数逆数
class QuaternionInv(Function):
    def forward(self, x0, x1):
        output = np.array([0, 0, 0, 0], np.single)
        x1_abs2 = x1[0]**2 + x1[1]**2 + x1[2]**2 + x1[3]**2
        output[0] = x0 * x1[0] / x1_abs2
        output[1] = -x0 * x1[1] / x1_abs2
        output[2] = -x0 * x1[2] / x1_abs2
        output[3] = -x0 * x1[3] / x1_abs2
        return output

# 実数割り算
class RealDiv(Function):
    def forward(self, x0, x1):
        return x0 / x1

class Rotation(Function):
    def forward(self, x, w):
        x0 = np.array([0, 0, 0, 0], np.single)
        w_abs = (w[0]**2 + w[1]**2 + w[2]**2 + w[3]**2)**0.5
        output = np.array([0, 0, 0, 0], np.single)
        x0[0] = w[0]*x[0] - w[1]*x[1] - w[2]*x[2] - w[3]*x[3]
        x0[1] = w[0]*x[1] + w[1]*x[0] + w[2]*x[3] - w[3]*x[2]
        x0[2] = w[0]*x[2] + w[2]*x[0] + w[3]*x[1] - w[1]*x[3]
        x0[3] = w[0]*x[3] + w[3]*x[0] + w[1]*x[2] - w[2]*x[1]
        
        # 丸め誤差
        # output[0] = (x0[0]*w[0] + x0[1]*w[1] + x0[2]*w[2] + x0[3]*w[3]) / w_abs
        output[0] = 0
        output[1] = (-x0[0]*w[1] + x0[1]*w[0] - x0[2]*w[3] + x0[3]*w[2]) / w_abs
        output[2] = (-x0[0]*w[2] + x0[2]*w[0] - x0[3]*w[1] + x0[1]*w[3]) / w_abs
        output[3] = (-x0[0]*w[3] + x0[3]*w[0] - x0[1]*w[2] + x0[2]*w[1]) / w_abs
        return output

# 活性化関数(tanh)
class ActivationFunction1(Function):
    def forward(self, x):
        output = np.array([0, 0, 0, 0], np.single)
        # 丸め誤差
        # output[0] = np.tanh(x[0])
        output[0] = 0
        output[1] = np.tanh(x[1])
        output[2] = np.tanh(x[2])
        output[3] = np.tanh(x[3])
        return output

class ActivationFunction2(Function):
    def forward(self, x):
        output = np.array([0, 0, 0, 0], np.single)
        x_abs = (x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2)**0.5
        output[0] = 0
        output[1] = np.tanh(x_abs) * x[1] / x_abs
        output[2] = np.tanh(x_abs) * x[2] / x_abs
        output[3] = np.tanh(x_abs) * x[3] / x_abs
        return output

# 活性化関数微分
class DiffAcrivationFunction(Function):
    def forward(self, x):
        output = np.array([0, 0, 0, 0], np.single)
        output[0] = 0
        if abs(x[1]) >= 4:
            output[1] = 0
        else:
            output[1] = 1 / (np.cosh(x[1]))**2
        if abs(x[2]) >= 4:
            output[2] = 0
        else:
            output[2] = 1 / (np.cosh(x[2]))**2
        if abs(x[3]) >= 4:
            output[3] = 0
        else:
            output[3] = 1 / (np.cosh(x[3]))**2
        return output

# np.arrayに変換
def as_array(x):
    if np.isscalar(x):
        return np.array(x, np.single)
    return x

# Variableに変換
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def quaternion_neg(x):
    return QuaterionNeg()(x)

def real_neg(x):
    return RealNeg()(x)

def scalar_multiply(x, s):
    s = as_array(s)
    return ScalarMultiply()(x, s)

def scalar_div(x, s):
    s = as_array(s)
    return ScalarDiv()(x, s)

def scalar_rdiv(x, s):
    s = as_array(s)
    return ScalarDiv()(s, x)

def conjugate(x):
    return Conjugate()(x)

def absolute_value(x):
    return AbsoluteValue()(x)
    
def quaternion_add(x0, x1):
    x1 = as_array(x1)
    return QuaternionAdd()(x0, x1)

def real_add(x0, x1):
    x1 = as_array(x1)
    return RealAdd()(x0, x1)

def quaternion_subtraction(x0, x1):
    x1 = as_array(x1)
    return QuaternionSubtraction()(x0, x1)

def quaternion_rsubtraction(x0, x1):
    x1 = as_array(x1)
    return QuaternionSubtraction()(x1, x0)

def real_subtraction(x0, x1):
    x1 = as_array(x1)
    return RealSubtraction()(x0, x1)

def real_rsubtraction(x0, x1):
    x1 = as_array(x1)
    return RealSubtraction()(x1, x0)

def outer_product(x0, x1):
    return OuterProduct()(x0, x1)

def inner_product(x0, x1):
    return InnerProduct()(x0, x1)

def hadamard_product(x0, x1):
    return HadamardProduct()(x0, x1)

def real_multiply(x0, x1):
    x1 = as_array(x1)
    return RealMultiply()(x0, x1)

def quaternion_div(x0, x1):
    x1 = as_array(x1)
    return QuaternionDiv()(x0, x1)

def quaternion_rdiv(x0, x1):
    x1 = as_array(x1)
    return QuaternionDiv()(x1, x0)

def quaternion_inv(x0, x1):
    x1 = as_array(x1)
    return QuaternionInv()(x0, x1)

def quaternion_rinv(x0, x1):
    x1 = as_array(x1)
    return QuaternionInv()(x1, x0)

def real_div(x0, x1):
    x1 = as_array(x1)
    return RealDiv()(x0, x1)

def real_rdiv(x0, x1):
    x1 = as_array(x1)
    return RealDiv()(x1, x0)

def rot(x, w):
    w = as_array(w)
    return Rotation()(x, w)

def tanh(x):
    return ActivationFunction1()(x)

def isotanh(x):
    return ActivationFunction2()(x)

def diff_tanh(x):
    return DiffAcrivationFunction()(x)

# -------------------
# classを継承していない関数
  
def input_rot(data, win):
    '''
    param data: 入力(1次元)
    param win: 入力四元数行列(1*N_x)
    '''
    output = copy.deepcopy(win)
    for i in range(win.size):
        output[i] = rot(data, win[i])
    return output

def reservoir_rot(w, x):
    '''
    param w: リザバー内重み四元数行列(N_x*N_x)
    param x: リザバー状態(1*N_x)
    '''
    N_x = len(x)
    output = np.full(N_x, Variable(np.array([0, 0, 0, 0])))
    for i in range(N_x):
        for j in range(N_x):
            output[i] += rot(x[j], w[j][i])
    return output

def output_rot(x, wout):
    '''
    param x: リザバー状態(1*N_x)
    param wout: 出力四元数行列(1*N_x)
    '''
    N_x = len(x)
    output = Variable(np.array([0, 0, 0 ,0]))
    for i in range(N_x):
        output += rot(x[i], wout[i])
    return output

def all_tanh(x):
    N_x = len(x)
    output = np.full(N_x, Variable(np.array([0, 0, 0, 0])))
    for i in range(N_x):
        output[i] = tanh(x[i])
    return output

def all_isotanh(x):
    N_x = len(x)
    output = np.full(N_x, Variable(np.array([0, 0, 0, 0])))
    for i in range(N_x):
        output[i] = isotanh(x[i])
    return output

def all_diff_tanh(x):
    N_x = len(x)
    output = np.full(N_x, Variable(np.array([0, 0, 0, 0])))
    for i in range(N_x):
        output[i] = diff_tanh(x[i])
    return output
