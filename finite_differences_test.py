import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from finite_differences import numeric_jacobian, numeric_jacobian_colwise


class Foo(object):
	def __init__(self, x):
		self.x = x

	class Atlas(object):
		@classmethod
		def dof(cls, x):
			return 1

		@classmethod
		def perturb(cls, f, v):
			return Foo(f.x + v[0])

		@classmethod
		def displacement(cls, f1, f2):
			return np.array([f2.x - f1.x])


class FiniteDifferencesTest(unittest.TestCase):
	def test_numeric_jacobian_scalar(self):
		f = lambda x: x**2
		assert_array_almost_equal(numeric_jacobian(f, 3.), [[6.]])

	def test_numeric_jacobian_vector_output(self):
		f = lambda x: np.array([x**2, 3.-x])
		assert_array_almost_equal(numeric_jacobian(f, 3.), [[6.], [-1.]])

	def test_numeric_jacobian_vector_input(self):
		f = lambda x: x[0] * 2.
		assert_array_almost_equal(numeric_jacobian(f, [3., 4.]), [[2., 0.]])

	def test_numeric_jacobian_vector_inout(self):
		f = lambda x: np.array((x[0] * 2., x[1]**2))
		assert_array_almost_equal(numeric_jacobian(f, [3., 4.]), [[2., 0.], [0., 8.]])

	def test_numeric_jacobian_custom_input(self):
		f = lambda f: f.x ** 2
		assert_array_almost_equal(numeric_jacobian(f, Foo(3.)), [[6.]])

	def test_numeric_jacobian_custom_output(self):
		f = lambda x: Foo(x ** 2)
		assert_array_almost_equal(numeric_jacobian(f, 3.), [[6.]])

	def test_numeric_jacobian_custom_inou(self):
		f = lambda f: Foo(f.x ** 2)
		assert_array_almost_equal(numeric_jacobian(f, Foo(3.)), [[6.]])


if __name__ == '__main__':
	unittest.main()
