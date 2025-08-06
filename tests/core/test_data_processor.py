import unittest
import pandas as pd
import numpy as np
import sys
import os

# By adding the project root to the path via a runner script,
# we can now use absolute imports from the 'src' directory.
from src.core.data_processor import process_data, clean_currency, validate_columns

class TestDataProcessing(unittest.TestCase):

    def test_clean_currency(self):
        self.assertEqual(clean_currency('Rp10.000'), 10000.0)
        self.assertEqual(clean_currency('Rp12.500,50'), 12500.5)
        self.assertEqual(clean_currency('5.000'), 5000.0)
        self.assertIsNone(clean_currency('invalid'))
        self.assertEqual(clean_currency(15000), 15000)

    def test_validate_columns(self):
        df = pd.DataFrame(columns=['id_transaksi', 'Waktu', 'nama_produk'])
        required = ['id_transaksi', 'waktu', 'nama_produk']
        try:
            validate_columns(df, required)
        except ValueError:
            self.fail("validate_columns() raised ValueError unexpectedly!")

        with self.assertRaises(ValueError):
            validate_columns(df, ['id_transaksi', 'waktu', 'harga'])

    def test_process_data(self):
        data = {
            'id_transaksi': [1, 2, 3],
            'waktu': ['2023-01-15', '2023-01-16', '2023-01-17'],
            'nama_pembeli': ['A', 'B', 'C'],
            'nama_kasir': ['X', 'Y', 'Z'],
            'nama_produk': ['Product A', 'Product B', 'Product A'],
            'kategori_produk': ['Cat1', 'Cat2', 'Cat1'],
            'harga_satuan': ['Rp10.000', 'Rp20.000', 'Rp10.000'],
            'jumlah': [1, 2, 3],
            'harga': ['Rp10.000', 'Rp40.000', 'Rp30.000'],
            'harga_setelah_pajak': ['Rp11.000', 'Rp44.000', 'Rp33.000'],
            'tipe_pesanan': ['Dine-in', 'Takeaway', 'Dine-in'],
            'metode_pembayaran': ['Cash', 'Credit', 'Cash'],
            'total_pembayaran': ['Rp11.000', 'Rp44.000', 'Rp33.000']
        }
        df = pd.DataFrame(data)
        processed_df = process_data(df)

        self.assertIn('tahun', processed_df.columns)
        self.assertIn('bulan', processed_df.columns)
        self.assertEqual(processed_df['tahun'].iloc[0], 2023)
        self.assertEqual(processed_df['bulan'].iloc[0], 1)
        self.assertEqual(processed_df['harga_satuan'].iloc[0], 10000.0)

if __name__ == '__main__':
    unittest.main()
