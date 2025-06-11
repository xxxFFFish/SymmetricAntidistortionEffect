import os
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Auxiliary Variates
CUTLINE = '--------------------------------'

# Constants
DATA_FILE_NAME = 'distortion_raw_data.csv'
PIXEL_PITCH = 0.006885 # Micrometer dimension. This value comes from 'distortion_raw_data.csv'
POLYFIT_DEGREE = 9
RESULT_SAVE_NAME_1 = 'polyfit_coeffs_with_tan_r_to_real_r'
RESULT_SAVE_NAME_2 = 'polyfit_coeffs_with_real_r_to_tan_r'

# Process Variates
g_args = None


def parse_argument():
    global g_args

    parser = argparse.ArgumentParser(description='fit data')
    parser.add_argument('-O', '--order', type=str, choices=('tan_r_to_real_r', 'real_r_to_tan_r'), help='fit order')

    g_args = parser.parse_args()

def get_distortion_raw_data():
    data_csv = pd.read_csv(DATA_FILE_NAME)

    field_r_array = np.array(data_csv.iloc[:, 4].values)
    real_x_array = np.array(data_csv.iloc[:, 7].values)
    real_y_array = np.array(data_csv.iloc[:, 8].values)

    tan_r_array = np.tan(np.radians(field_r_array))
    real_r_array = np.sqrt(real_x_array**2 + real_y_array**2)
    real_r_array /= PIXEL_PITCH

    return tan_r_array, real_r_array

def fit_with_polyfit(in_array, out_array):
    return np.polyfit(in_array, out_array, POLYFIT_DEGREE)

def show_result_figure(x_array, y_array, fit_coeffs, x_label='x array', y_label='y array', save_name=''):
    plt.figure(figsize=(8,8))
    plt.scatter(x_array, y_array, c='red', s=4, marker='o')

    poly = np.poly1d(fit_coeffs)
    poly_x_array = np.linspace(x_array.min(), x_array.max(), 1000)
    poly_y_array = poly(poly_x_array)
    plt.plot(poly_x_array, poly_y_array, 'g-', linewidth=2)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.7)

    if save_name.strip():
        plt.savefig(save_name + '.png')
    else:
        plt.show()


if __name__ == '__main__':
    print(CUTLINE)
    print('Start Fit Data Process')
    start_time = time.time()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parse_argument()

    tan_r_array, real_r_array = get_distortion_raw_data()

    # Fit tan_r to real_r curve and save
    if g_args.order == 'tan_r_to_real_r':
        coeffs = fit_with_polyfit(tan_r_array, real_r_array)
        np.save(RESULT_SAVE_NAME_1 + '.npy', coeffs) # Save as numpy array file

        coeffs_df = pd.DataFrame(coeffs[::-1]) # Reverse order when save as csv
        coeffs_df.columns = ['Coeffs']
        coeffs_df.to_csv(RESULT_SAVE_NAME_1 + '.csv', index=False)

        show_result_figure(tan_r_array, real_r_array, coeffs, 'tan r', 'real r', RESULT_SAVE_NAME_1)

    # Fit real_r to tan_r curve and save
    if g_args.order ==  'real_r_to_tan_r':
        coeffs = fit_with_polyfit(real_r_array, tan_r_array)
        np.save(RESULT_SAVE_NAME_2 + '.npy', coeffs) # Save as numpy array file

        coeffs_df = pd.DataFrame(coeffs[::-1]) # Reverse order when save as csv
        coeffs_df.columns = ['Coeffs']
        coeffs_df.to_csv(RESULT_SAVE_NAME_2 + '.csv', index=False)

        show_result_figure(real_r_array, tan_r_array, coeffs, 'real r', 'tan r', RESULT_SAVE_NAME_2) # Draw curve and save as figure

    consume_time = time.time() - start_time
    print('End Fit Data Process')
    print('Consume time: %.4fs' % (consume_time))
    print(CUTLINE)
