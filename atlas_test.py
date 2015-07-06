import unittest
import numpy as np
from numericaltesting import assert_arrays_almost_equal

from .atlas import atlas_for, CartesianAtlas, ScalarAtlas


class Foo(object):
	Atlas = "abc"

class AtlasTest(unittest.TestCase):
	def test_atlas_for(self):
		f = Foo()
		self.assertEqual(atlas_for(f), "abc")
		self.assertIs(atlas_for(1), ScalarAtlas)
		self.assertIs(atlas_for(np.arange(3)), CartesianAtlas)

	def test_cartesian_atlas(self):
		a = np.array([1, 2, 3])
		assert_arrays_almost_equal(CartesianAtlas.perturb(a, [1, 0, -1]), [2, 2, 2])
		assert_arrays_almost_equal(CartesianAtlas.displacement(a, [2, 2, 2]), [1, 0, -1])

	def test_scalar_atlas(self):
		self.assertEqual(ScalarAtlas.perturb(3, [-5]), -2)
		assert_arrays_almost_equal(ScalarAtlas.displacement(2, 6), [4])


if __name__ == '__main__':
	unittest.main()
