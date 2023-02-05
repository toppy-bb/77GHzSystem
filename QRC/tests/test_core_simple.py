import unittest
import numpy as np
import sys
sys.path.append('../')

from quaternion_calculation import *
# from QRC.core_simple import *

x0 = Variable(np.array(3.0))
x1 = 4
x2 = Variable(np.array([1,2,3,4]))
x3 = Variable(np.array([5,6,7,8]))

class TestAdd(unittest.TestCase):
    def test_add(self):        
        self.assertEqual((x0+x1).data, np.array(7.0))
        self.assertEqual((x1+x0).data, np.array(7.0))
        self.assertTrue(np.array_equal((x2+x3).data, np.array([6,8,10,12])))

class TestSub(unittest.TestCase):
    def test_sub(self):
        self.assertEqual((x0-x1).data, np.array(-1))
        self.assertEqual((x1-x0).data, np.array(1.0))
        self.assertTrue(np.array_equal((x2-x3).data, np.array([-4,-4,-4,-4])))

class TestMul(unittest.TestCase):    
    def test_mul(self):
        self.assertEqual((x0*x1).data, np.array(12))
        self.assertEqual((x1*x0).data, np.array(12.0))
        self.assertTrue(np.array_equal((x0*x2).data, np.array([3,6,9,12])))
        self.assertTrue(np.array_equal((x2*x0).data, np.array([3,6,9,12])))
        self.assertTrue(np.array_equal((x1*x2).data, np.array([4,8,12,16])))
        self.assertTrue(np.array_equal((x2*x1).data, np.array([4,8,12,16])))
        self.assertTrue(np.array_equal((x2*x3).data, np.array([-60,12,30,24])))

class TestDiv(unittest.TestCase):    
    def test_div(self):
        self.assertEqual((x0/x1).data, np.array(0.75))
        self.assertEqual((x1/x0).data, np.array(4/3, dtype='float32'))
        self.assertTrue(np.array_equal((x0/x2).data, np.array([0.1,-0.2,-0.3,-0.4], dtype='float32')))
        self.assertTrue(np.array_equal((x2/x0).data, np.array([1/3,2/3,1,4/3], dtype='float32')))
        self.assertTrue(np.array_equal((x1/x2).data, np.array([2/15,-4/15,-6/15,-8/15], dtype='float32')))
        self.assertTrue(np.array_equal((x2/x1).data, np.array([0.25,0.5,0.75,1], dtype='float32')))
        self.assertTrue(np.array_equal((x3/x2).data, np.array([7/3,-8/30,0,-16/30], dtype='float32')))
