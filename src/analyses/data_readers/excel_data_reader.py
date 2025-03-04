import os
from pathlib import Path

import pandas as pd


class ExcelDataReader:

    BASE_DIR = '/home/connorlab/Documents/GitHub/Julie/'

    def __init__(self, subfolder, file_name):
        self.subfolder = subfolder
        self.file_name = file_name
        self.file_path = Path(self.BASE_DIR) / self.subfolder / self.file_name

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self.xl = pd.ExcelFile(self.file_path)

    def get_raw_data(self):
        return self.xl

    def get_sheet_by_name(self, sheet_name):
        return self.xl.parse(sheet_name)

    def get_first_sheet(self):
        sheet_names = self.xl.sheet_names
        return self.xl.parse(sheet_names[0])

    def get_last_sheet(self):
        sheet_names = self.xl.sheet_names
        return self.xl.parse(sheet_names[-1])