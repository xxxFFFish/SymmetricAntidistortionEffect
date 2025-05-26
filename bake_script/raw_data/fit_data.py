import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Auxiliary Variates
CUTLINE = '--------------------------------'

# Constants
DATA_FILE_NAME = 'distortion_raw_data.csv'
POLYFIT_DEGREE = 9
RESULT_SAVE_NAME_1 = 'polyfit_coeffs_with_tan_r_to_real_r'
RESULT_SAVE_NAME_2 = 'polyfit_coeffs_with_real_r_to_tan_r'

# Process Variates
g_fit_order = 0 # 1 is tan_r to real_r, 2 is real_r to tan_r


def parse_cmd():
    global g_fit_order

    order = ''

    for cmd in sys.argv[1:]: # ignore argv[0]
        cmd_collection = cmd.split('=')
        if len(cmd_collection) > 1:
            cmd_collection[0].lower()
            if cmd_collection[0] == 'order' or cmd_collection[0] == 'direction':
                order = cmd_collection[1]
    
    if order == 'tan_r_to_real_r':
        g_fit_order = 1
    elif order == 'real_r_to_tan_r':
        g_fit_order = 2


def get_distortion_raw_data():
    data_csv = pd.read_csv(DATA_FILE_NAME)

    field_r_array = np.array(data_csv.iloc[:, 4].values)
    real_x_array = np.array(data_csv.iloc[:, 7].values)
    real_y_array = np.array(data_csv.iloc[:, 8].values)

    tan_r_array = np.tan(np.radians(field_r_array))
    real_r_array = np.sqrt(real_x_array**2 + real_y_array**2)

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

    if len(sys.argv) > 1:
        parse_cmd()

    tan_r_array, real_r_array = get_distortion_raw_data()

    # Fit tan_r to real_r curve and save
    if g_fit_order == 0 or g_fit_order == 1:
        coeffs = fit_with_polyfit(tan_r_array, real_r_array)
        np.save(RESULT_SAVE_NAME_1 + '.npy', coeffs) # Save as numpy array file

        coeffs_df = pd.DataFrame(coeffs[::-1]) # Reverse order when save as csv
        coeffs_df.columns = ['Coeffs']
        coeffs_df.to_csv(RESULT_SAVE_NAME_1 + '.csv', index=False)

        show_result_figure(tan_r_array, real_r_array, coeffs, 'tan r', 'real r', RESULT_SAVE_NAME_1)

    # Fit real_r to tan_r curve and save
    if g_fit_order == 0 or g_fit_order == 2:
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
