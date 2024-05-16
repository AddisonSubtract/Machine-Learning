"""
# Example of reading data from a csv file and getting it into a numpy matrix.
"""

import os           # can be useful for dealing with file paths
import csv          # useful for reading a csv file; the pandas package is also helpful for this
import numpy as np

def read_housing_csv_file(fpath):

    data = []  # this list will be used to store each observation from the file

    with open(fpath, 'r') as infile:
        freader = csv.reader(infile)

        header = True

        for row in freader:
            
            if header:
                # This is just a silly way to ignore the column names in the file
                header = False
                
            else:
                id_value = int(row[0])
                lot_area = float(row[1])
                house_price = float(row[2])
                
                data.append([id_value, lot_area, house_price])

    data_as_numpy_matrix = np.array(data)

    return data_as_numpy_matrix


if __name__ == "__main__":
    fpath = 'ml_a1_data.csv'
    data_matrix = read_housing_csv_file(fpath)
    print('The data is now in a ' + str(data_matrix.shape[0]) + ' by ' + str(data_matrix.shape[1]) + ' matrix.')

    # The first row of the matrix:
    first_row = data_matrix[0]

    # The element in the second row, second column:
    element = data_matrix[1, 1]

    # The second column of the matrix:
    second_column = data_matrix[:, 1]  # the colon here means "all"
    
    
