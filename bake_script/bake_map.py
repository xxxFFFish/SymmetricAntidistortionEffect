import os
import sys
import time
import subprocess
import math
import numpy as np
import gzip
import cv2


# Auxiliary Variates
CUTLINE = '--------------------------------' + '--------------------------------'

# Constants
POLYFIT_COEFFS_FILE_PATH = 'raw_data'
POLYFIT_COEFFS_FILE_NAME = 'polyfit_coeffs_with_real_r_to_tan_r.npy'
FIT_SCRIPT = 'fit_data.py'
SAVE_MAP_NAME = 'map' # Final name is SAVE_MAP_NAME + g_map_model + g_texture_model
SAVE_MAP_FOLDER = 'map_texture'

# Setting Variates
g_screen_width = 1920
g_screen_height = 1080
g_pixel_scale = 0.01 # Micrometer dimension. It must be carried out in conjunction with distortion model.
g_map_model = 'keep_center' # keep_center, expand_edge, custom
g_texture_model = 'minisize' # minisize, balance, free
g_split = False
g_save_format = 'png' # png, gz

# Process Variates
g_center_shift_x = 0.0
g_center_shift_y = 0.0
g_tan_factor = 0.1


def parse_cmd():
    global g_screen_width
    global g_screen_height
    global g_pixel_scale
    global g_map_model
    global g_texture_model
    global g_split
    global g_save_format

    width = 0
    height = 0
    scale = 0.0
    m_model = ''
    t_model = ''
    s_format = ''

    for cmd in sys.argv[1:]: # ignore argv[0]
        cmd_collection = cmd.split('=')
        if len(cmd_collection) > 1:
            cmd_collection[0].lower()
            if cmd_collection[0] == 'screen_width' or cmd_collection[0] == 'width':
                width = int(cmd_collection[1])
            elif cmd_collection[0] == 'screen_height' or cmd_collection[0] == 'height':
                height = int(cmd_collection[1])
            elif cmd_collection[0] == 'pixel_scale' or cmd_collection[0] == 'scale':
                scale = float(cmd_collection[1])
            elif cmd_collection[0] == 'map_model' or cmd_collection[0] == 'm_model':
                m_model = cmd_collection[1]
            elif cmd_collection[0] == 'texture_model' or cmd_collection[0] == 't_model':
                t_model = cmd_collection[1]
            elif cmd_collection[0] == 'split' and cmd_collection[1] == '1':
                g_split = True
            elif cmd_collection[0] == 'texture_format' or cmd_collection[0] == 'save_format':
                s_format = cmd_collection[1]

    if width > 0 and height > 0:
        g_screen_width = width
        g_screen_height = height

    if scale > 0.0:
        g_pixel_scale = scale

    if m_model == 'keep_center' or m_model == 'expand_edge' or m_model == 'custom':
        g_map_model = m_model

    if t_model == 'minisize' or t_model == 'balance' or t_model == 'free':
        g_texture_model = t_model
    
    if s_format == 'png' or m_model == 'gz':
        g_save_format = s_format

def get_polyfit_coeffs():
    file_path = POLYFIT_COEFFS_FILE_PATH + '/' + POLYFIT_COEFFS_FILE_NAME

    if os.path.isfile(file_path):
        coeffs = np.load(file_path)
        return coeffs
    else:
        # Find none polyfit coeffs file, need to execute fit script.
        subprocess.run(['python', FIT_SCRIPT, 'order=real_r_to_tan_r'], cwd=POLYFIT_COEFFS_FILE_PATH)

    if os.path.isfile(file_path):
        coeffs = np.load(file_path)
        return coeffs
    else:
        print('Get %s error' % (file_path))
        return np.array([])

def get_r_from_pixel(x: int, y: int):
    _x = float(x) - g_center_shift_x
    _y = float(y) - g_center_shift_y
    return g_pixel_scale*math.sqrt(_x*_x + _y*_y)

def get_tan_factor_based_on_keeping_center(coeffs: np.ndarray):
    poly_deriv = np.poly1d(coeffs).deriv(m=1)
    deriv_center = poly_deriv(0)

    # delta_tan_r_center / delta_real_r_center = deriv_center
    # delta_tan_r_center = delta_r_center_map / tan_factor
    # Keep center factor is 1: delta_r_center_map / delta_real_r_center = 1
    # tan_factor = 1 / deriv_center

    return 1.0 / deriv_center

def get_tan_factor_based_on_expanding_edge(coeffs: np.ndarray):
    r_edge = 0.0

    if g_screen_width > g_screen_height:
        r_edge = get_r_from_pixel(g_screen_width / 2, 0)
    else:
        r_edge = get_r_from_pixel(0, g_screen_height / 2)

    poly = np.poly1d(coeffs)
    tan_r_edge = poly(r_edge)

    # tan_r_edge = r_edge_map / tan_factor
    # expand edge to screen edge: r_edge_map = r_edge
    # tan_factor = r_edge / tan_r_edge

    return r_edge / tan_r_edge

def get_tan_factor(coeffs: np.ndarray):
    if g_map_model == 'keep_center':
        return get_tan_factor_based_on_keeping_center(coeffs)
    elif g_map_model == 'expand_edge':
        return get_tan_factor_based_on_expanding_edge(coeffs)
    elif g_map_model == 'custom':
        return 0.1 # Custom map model is TBD
    else:
        return 0.1

def get_antidistortion_map_texture_based_on_minisize(coeffs: np.ndarray):
    # Only need to calculate 1/4 area by symmetry
    texture_width = g_screen_width >> 1
    texture_height = g_screen_height >> 1

    texture = np.zeros([texture_height, texture_width, 4], dtype = np.uint8, order = 'C')
    poly = np.poly1d(coeffs)

    # Filter 45 degree symmetry direction area
    filter_refrence = texture_width - texture_height

    for y in range(0, texture_height):
        for x in range(0, texture_width):
            if filter_refrence >= 0 and x > y + filter_refrence:
                continue

            if filter_refrence <= 0 and x - filter_refrence < y:
                continue

            r_preset = get_r_from_pixel(x, y)
            r_map = g_tan_factor*poly(r_preset)
            map_bytes = np.float32(r_map / r_preset).view(np.uint32).tobytes()
            texture[y, x, 0] = map_bytes[0]
            texture[y, x, 1] = map_bytes[1]
            texture[y, x, 2] = map_bytes[2]
            texture[y, x, 3] = map_bytes[3]

    return texture

def get_antidistortion_map_texture_based_on_balance(coeffs: np.ndarray):
    # Only need to calculate 1/4 area by symmetry
    texture_width = g_screen_width >> 1
    texture_height = g_screen_height >> 1

    # Store the UV of X axis and Y axis independently
    texture = np.zeros([texture_height, texture_width << 1, 4], dtype = np.uint8, order = 'C')
    poly = np.poly1d(coeffs)

    for y in range(0, texture_height):
        for x in range(0, texture_width):
            if g_split:
                r_preset = get_r_from_pixel(2*x, y)
                r_map = g_tan_factor*poly(r_preset)
                ratio = r_map / r_preset

                uv = float(2*x) / float(g_screen_width - 1)
                uv = ((uv - 0.5)*ratio + 0.5)*0.5
                if uv > 0.5:
                    uv += 0.5
            else:
                r_preset = get_r_from_pixel(x, y)
                r_map = g_tan_factor*poly(r_preset)
                ratio = r_map / r_preset

                uv = float(x) / float(g_screen_width - 1)
                uv = (uv - 0.5)*ratio + 0.5

            uv_bytes = np.float32(uv).view(np.uint32).tobytes()
            texture[y, x, 0] = uv_bytes[0]
            texture[y, x, 1] = uv_bytes[1]
            texture[y, x, 2] = uv_bytes[2]
            texture[y, x, 3] = uv_bytes[3]

            uv = float(y) / float(g_screen_height - 1)
            uv = (uv - 0.5)*ratio + 0.5
            uv_bytes = np.float32(uv).view(np.uint32).tobytes()
            _x = x + texture_width
            texture[y, _x, 0] = uv_bytes[0]
            texture[y, _x, 1] = uv_bytes[1]
            texture[y, _x, 2] = uv_bytes[2]
            texture[y, _x, 3] = uv_bytes[3]

    return texture

def get_antidistortion_map_texture_based_on_free(coeffs: np.ndarray):
    # Only need to calculate 1/4 area by symmetry
    texture_width = g_screen_width >> 1
    texture_height = g_screen_height >> 1

    # Store the UV of X axis and Y axis completely
    texture = np.zeros([g_screen_height, g_screen_width << 1, 4], dtype = np.uint8, order = 'C')
    poly = np.poly1d(coeffs)

    for y in range(0, texture_height):
        for x in range(0, texture_width):
            if g_split:
                r_preset = get_r_from_pixel(2*x, y)
                r_map = g_tan_factor*poly(r_preset)
                ratio = r_map / r_preset

                uv = float(2*x) / float(g_screen_width - 1)
                uv = ((uv - 0.5)*ratio + 0.5)*0.5
                if uv > 0.5:
                    uv += 0.5
            else:
                r_preset = get_r_from_pixel(x, y)
                r_map = g_tan_factor*poly(r_preset)
                ratio = r_map / r_preset

                uv = float(x) / float(g_screen_width - 1)
                uv = (uv - 0.5)*ratio + 0.5

            # Store UV.x
            # Top-Left area
            uv_bytes = np.float32(uv).view(np.uint32).tobytes()
            texture[y, x, 0] = uv_bytes[0]
            texture[y, x, 1] = uv_bytes[1]
            texture[y, x, 2] = uv_bytes[2]
            texture[y, x, 3] = uv_bytes[3]

            # Bottom-Left area
            _y = g_screen_height - 1 - y
            texture[_y, x, 0] = uv_bytes[0]
            texture[_y, x, 1] = uv_bytes[1]
            texture[_y, x, 2] = uv_bytes[2]
            texture[_y, x, 3] = uv_bytes[3]

            # Bottom-Right area
            uv = 1.0 - uv
            uv_bytes = np.float32(uv).view(np.uint32).tobytes()
            _x = g_screen_width - 1 - x
            texture[_y, _x, 0] = uv_bytes[0]
            texture[_y, _x, 1] = uv_bytes[1]
            texture[_y, _x, 2] = uv_bytes[2]
            texture[_y, _x, 3] = uv_bytes[3]

            # Top-Right area
            texture[y, _x, 0] = uv_bytes[0]
            texture[y, _x, 1] = uv_bytes[1]
            texture[y, _x, 2] = uv_bytes[2]
            texture[y, _x, 3] = uv_bytes[3]

            # Store UV.y
            # Top-Left area
            uv = float(y) / float(g_screen_height - 1)
            uv = (uv - 0.5)*ratio + 0.5
            uv_bytes = np.float32(uv).view(np.uint32).tobytes()
            _x = x + g_screen_width
            texture[y, _x, 0] = uv_bytes[0]
            texture[y, _x, 1] = uv_bytes[1]
            texture[y, _x, 2] = uv_bytes[2]
            texture[y, _x, 3] = uv_bytes[3]

            # Top-Right area
            _x = g_screen_width - 1 - _x
            texture[y, _x, 0] = uv_bytes[0]
            texture[y, _x, 1] = uv_bytes[1]
            texture[y, _x, 2] = uv_bytes[2]
            texture[y, _x, 3] = uv_bytes[3]

            # Bottom-Right area
            uv = 1.0 - uv
            uv_bytes = np.float32(uv).view(np.uint32).tobytes()
            texture[_y, _x, 0] = uv_bytes[0]
            texture[_y, _x, 1] = uv_bytes[1]
            texture[_y, _x, 2] = uv_bytes[2]
            texture[_y, _x, 3] = uv_bytes[3]

            # Bottom-Left area
            _x = x + g_screen_width
            texture[_y, _x, 0] = uv_bytes[0]
            texture[_y, _x, 1] = uv_bytes[1]
            texture[_y, _x, 2] = uv_bytes[2]
            texture[_y, _x, 3] = uv_bytes[3]

    return texture

def get_antidistortion_map_texture(coeffs: np.ndarray):
    if g_texture_model == 'minisize':
        return get_antidistortion_map_texture_based_on_minisize(coeffs)
    elif g_texture_model == 'balance':
        return get_antidistortion_map_texture_based_on_balance(coeffs)
    elif g_texture_model == 'free':
        return get_antidistortion_map_texture_based_on_free(coeffs)
    else:
        return np.array([])

def save_texture(file_name: str, texture: np.ndarray):
    if not os.path.exists(SAVE_MAP_FOLDER):
        os.makedirs(SAVE_MAP_FOLDER)

    file_full_name = file_name + '_' + g_map_model + '_' + g_texture_model
    if g_texture_model != 'minisize' and g_split:
        file_full_name += '_split'

    # Save texture as compressed bin file
    if g_save_format == 'gz':
        save_file_name = SAVE_MAP_FOLDER + '/' + file_full_name + '.gz'

        with gzip.open(save_file_name, 'wb') as f:
            f.write(texture)
        
        print('Save %s success' % (save_file_name))

    # Save texture as png
    if g_save_format == 'png':
        save_file_name = SAVE_MAP_FOLDER + '/' + file_full_name + '.png'
        bgra_texture = cv2.cvtColor(texture, cv2.COLOR_RGBA2BGRA) # Default color order in OpenCV is BGR
        cv2.imwrite(save_file_name, bgra_texture)

        print('Save %s success' % (save_file_name))


if __name__ == '__main__':
    print(CUTLINE)
    print('Start Bake Antidistortion Map')
    start_time = time.time()

    if len(sys.argv) > 1:
        parse_cmd()

    print('Map resolution: %dx%d' % (g_screen_width, g_screen_height))
    print('Pixel scale: %f' % (g_pixel_scale))
    print('Map model: %s' % (g_map_model))
    print('Texture model: %s' % (g_texture_model))
    print('Split: %s' % (g_split))
    print('Save format: %s' % (g_save_format))

    # Preset parameters
    g_center_shift_x = (g_screen_width - 1)*0.5
    g_center_shift_y = (g_screen_height - 1)*0.5

    polyfit_coeffs = get_polyfit_coeffs()
    if polyfit_coeffs.size > 1:
        g_tan_factor = get_tan_factor(polyfit_coeffs)
        print('Map tan factor: %f' % (g_tan_factor))

        map_texture = get_antidistortion_map_texture(polyfit_coeffs)
        if map_texture.size > 0:
            save_texture(SAVE_MAP_NAME, map_texture)
        else:
            print('Bake map texture failed')

    consume_time = time.time() - start_time
    print('End Bake Antidistortion Map')
    print('Consume time: %.4fs' % (consume_time))
    print(CUTLINE)
