#/bin/python
# %%[markdown]
# Copyright (2023) David A. Magezi
# %%

# %%[markdown]
# Bibliotheken
import numpy as np
import pandas as pd

# %%
class  Results():
    def __init__(self,to_csv=False):
        self.file_column_name='filename'
        self.numeric_column_names = ['center_x_mm','center_y_mm','semi_axes_a_mm','semi_axes_b_mm','angle_rad']
        self.file_names_list = []
        self.number_numeric_columns = len(self.numeric_column_names)
        self.numeric = []
        self.formatter = '{:6.6f}'
        self.to_csv = to_csv

    def nextFile(self,file_name,result):
        if (np.size(result) == self.number_numeric_columns):
            self.file_names_list.append(file_name)
            self.numeric.append(result)
        else:
            print("ERROR: results have incorrect dimensions")

    def print(self):
        if (len(self.numeric) == len(self.file_names_list)):
            numeric_list=[]
            for iRow in self.numeric:
                nextrow=[]
                for iCol in np.arange(self.number_numeric_columns):
                    nextrow.append(self.formatter.format(iRow[iCol]))
                numeric_list.append(nextrow)
            df = pd.DataFrame(data=numeric_list,columns=self.numeric_column_names)
            file_names=pd.Series(data=self.file_names_list,name='filename')
            results_df = pd.concat([file_names,df],axis=1)
            if self.to_csv:
                results_df.to_csv(path_or_buf="challenge_results.csv",index=False)
            else:
                print(results_df)
        else:
            print("ERROR: number of filenames does not match rows of results")
