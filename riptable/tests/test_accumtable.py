import unittest
import riptable as rt


class AccumTable_Test(unittest.TestCase):
    def test_accum_cols(self):

        num_rows = 10
        data = rt.Dataset(
            {
                'Symb': rt.Cat(['A', 'B'] * int(num_rows / 2)),
                'Count': rt.full(num_rows, 1.0),
                'PlusMinus': [1.0, -1.0]
                * int(num_rows / 2),  # Added to handle edge case of zero footer
            }
        )

        accum = rt.accum_cols(
            data.Symb, [data.Count, data.PlusMinus], ['Count', 'PlusMinus']
        )
        accum_expected = rt.Dataset(
            {'Symb': ['A', 'B'], 'Count': [5.0, 5.0], 'PlusMinus': [5.0, -5.0]}
        )
        accum_expected.footer_set_values(
            'Total', {'Symb': 'Total', 'Count': 10.0, 'PlusMinus': 0.0}
        )
        self.assertTrue((accum == accum_expected).all(axis=None))

    def test_accum_cols_multikey(self):
        num_rows = 12
        data = rt.Dataset(
            {
                'Symb': rt.Cat(['A', 'B'] * int(num_rows / 2)),
                'Exch': rt.Cat(['X', 'Y', 'Y', 'X'] * int(num_rows / 4)),
                'Count': rt.full(num_rows, 1.0),
                'PlusMinus': [1.0, -1.0] * int(num_rows / 2),
            }
        )
        data.MultiKeyCat = rt.Cat([data.Symb, data.Exch])

        accum = rt.accum_cols(
            data.MultiKeyCat, [data.Count, data.PlusMinus], ['Count', 'PlusMinus']
        )
        accum_expected = rt.Dataset(
            {
                'Symb': ['A', 'B', 'A', 'B'],
                'Exch': ['X', 'Y', 'Y', 'X'],
                'Count': [3.0, 3.0, 3.0, 3.0],
                'PlusMinus': [3.0, -3.0, 3.0, -3.0],
            }
        )
        accum_expected.footer_set_values(
            'Total', {'Exch': 'Total', 'Count': 12.0, 'PlusMinus': 0.0}
        )

        self.assertTrue((accum == accum_expected).all(axis=None))

    # When a raw FA is passed as the pointers instead of Categorical
    def test_accum_cols_noncat(self):
        num_rows = 10
        pointer = rt.FA([0, 1] * int(num_rows / 2))
        count = rt.full(num_rows, 1.0)

        accum = rt.accum_cols(pointer, count)
        accum_expected = rt.Dataset({'YLabel': [0, 1], 'col0': [5.0, 5.0]})
        accum_expected.footer_set_values('Total', {'YLabel': 'Total', 'col0': 10.0})

        self.assertTrue((accum == accum_expected).all(axis=None))

    # Test basic accum_ratiop
    def test_accum_ratiop(self):

        num_rows = 12
        data = rt.Dataset(
            {
                'Symb': rt.Cat(['A', 'A', 'A', 'B'] * int(num_rows / 4)),
                'Exch': rt.Cat(['Z', 'Z', 'X', 'X'] * int(num_rows / 4)),
                'Count': rt.full(num_rows, 1.0),
            }
        )

        # Invalid input
        with self.assertRaises(
            ValueError, msg=f'Failed to raise an error when passing invalid norm_by arg'
        ):
            rt.accum_ratiop(data.Symb, data.Exch, data.Count, norm_by='z')

        # Ratio within total
        accum = rt.accum_ratiop(data.Symb, data.Exch, data.Count, norm_by='T')
        accum_expected = rt.Dataset(
            {
                'Symb': ['A', 'B'],
                'X': [25.0, 25.0],
                'Z': [50.0, 0.0],
                'TotalRatio': [75.0, 25.0],
                'Total': [9.0, 3.0],
            }
        )
        accum_expected.footer_set_values(
            'TotalRatio',
            {'Symb': 'TotalRatio', 'X': 50.0, 'Z': 50.0, 'TotalRatio': 100.0},
        )
        accum_expected.footer_set_values(
            'Total', {'Symb': 'Total', 'X': 6.0, 'Z': 6.0, 'Total': 12.0}
        )
        self.assertTrue((accum == accum_expected).all(axis=None))

        # Ratio within columns
        accum = rt.accum_ratiop(data.Symb, data.Exch, data.Count, norm_by='c')
        accum_expected = rt.Dataset(
            {
                'Symb': ['A', 'B'],
                'X': [50.0, 50.0],
                'Z': [100.0, 0.0],
                'TotalRatio': [75.0, 25.0],
                'Total': [9.0, 3.0],
            }
        )
        accum_expected.footer_set_values(
            'TotalRatio',
            {'Symb': 'TotalRatio', 'X': 100.0, 'Z': 100.0, 'TotalRatio': 100.0},
        )
        accum_expected.footer_set_values(
            'Total', {'Symb': 'Total', 'X': 6.0, 'Z': 6.0, 'Total': 12.0}
        )
        self.assertTrue((accum == accum_expected).all(axis=None))


if __name__ == "__main__":
    tester = unittest.main()
