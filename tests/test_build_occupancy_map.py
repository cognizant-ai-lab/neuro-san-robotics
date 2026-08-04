import unittest

import numpy as np

from maps.build_occupancy_map import _compact_structures


class CompactStructureTests(unittest.TestCase):

    def test_retains_chair_sized_closed_shape(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[3, 3:7] = True
        mask[6, 3:7] = True
        mask[3:7, 3] = True
        mask[3:7, 6] = True

        kept = _compact_structures(
            mask,
            minimum_area_cells=6,
            minimum_span_cells=3,
        )

        self.assertTrue(np.array_equal(kept, mask))

    def test_rejects_narrow_text_like_stroke(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[3:8, 4:6] = True

        kept = _compact_structures(
            mask,
            minimum_area_cells=6,
            minimum_span_cells=3,
        )

        self.assertFalse(kept.any())

    def test_keeps_separate_furniture_without_joining_plan_noise(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[2:6, 2:6] = True
        mask[10:14, 10:14] = True
        mask[7, 7] = True

        kept = _compact_structures(
            mask,
            minimum_area_cells=6,
            minimum_span_cells=3,
        )

        self.assertTrue(kept[2:6, 2:6].all())
        self.assertTrue(kept[10:14, 10:14].all())
        self.assertFalse(kept[7, 7])


if __name__ == "__main__":
    unittest.main()
