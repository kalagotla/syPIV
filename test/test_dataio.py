import unittest


class TestDataIO(unittest.TestCase):
    def test_dataio(self):
        from src.dataio import DataIO
        data = DataIO('../data/0.1/Z281.*')
        data.read_data()

        print("Data shape is ", data.df.shape)

        self.assertEqual(data.df.shape[1], 9)
        self.assertEqual(data.psize, 281)


if __name__ == '__main__':
    unittest.main()
