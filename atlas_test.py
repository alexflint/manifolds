import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

import atlas


class Foo(object):
	Atlas = "abc"

class AtlasTest(unittest.TestCase):
	def test_atlas_for(self):
		f = Foo()
		self.assertEqual(atlas.atlas_for(f), "abc")
		self.assertIs(atlas.atlas_for(1), atlas.ScalarAtlas)
		self.assertIs(atlas.atlas_for(np.arange(3)), atlas.CartesianAtlas)

	def test_cartesian_atlas(self):
		a = np.array([1, 2, 3])
		assert_array_almost_equal(atlas.CartesianAtlas.perturb(a, [1, 0, -1]), [2, 2, 2])
		assert_array_almost_equal(atlas.CartesianAtlas.displacement(a, [2, 2, 2]), [1, 0, -1])

	def test_scalar_atlas(self):
		self.assertEqual(atlas.ScalarAtlas.perturb(3, [-5]), -2)
		assert_array_almost_equal(atlas.ScalarAtlas.displacement(2, 6), [4])


if __name__ == '__main__':
	unittest.main()
