import os
import time
import subprocess
import shutil

# Auxiliary Variates
CUTLINE = '--------------------------------' + '--------------------------------' + '--------------------------------'

# Constants
BAKE_SCRIPT = 'bake_map.py'
SOURCE_MAP_FOLDER = 'map_texture'
DESTINATION_MAP_FOLDER = '../godot-app/map_texture'

# Setting Variates
g_screen_width = '1920'
g_screen_height = '1080'
g_pixel_scale = '0.006885' # Micrometer dimension.
g_save_format = 'png'

# Process Variates
g_bake_count = 0


def run_bake(m_model: str, t_model: str, split: bool):
    global g_bake_count

    subprocess.run([
        'python',
        BAKE_SCRIPT,
        'width=' + g_screen_width,
        'height=' + g_screen_height,
        'scale=' + g_pixel_scale,
        'm_model=' + m_model,
        't_model=' + t_model,
        'split=' + '1' if split else '0',
        'save_format=' + g_save_format
    ])

    g_bake_count += 1
    print('Complete baking map with %s %s%s' % (m_model, t_model, ' split' if split else ''))

def run_bake_all():
    run_bake('keep_center', 'minisize', False)
    run_bake('expand_edge', 'minisize', False)

    run_bake('keep_center', 'balance', False)
    run_bake('keep_center', 'balance', True)
    run_bake('expand_edge', 'balance', False)
    run_bake('expand_edge', 'balance', True)

    run_bake('keep_center', 'free', False)
    run_bake('keep_center', 'free', True)
    run_bake('expand_edge', 'free', False)
    run_bake('expand_edge', 'free', True)

def copy_map_texture():
    if not os.path.exists(SOURCE_MAP_FOLDER):
        print('Souce path %s is invalid' % (SOURCE_MAP_FOLDER))
        return

    if not os.path.exists(DESTINATION_MAP_FOLDER):
        os.makedirs(DESTINATION_MAP_FOLDER)
    
    copy_count = 0

    for item in os.listdir(SOURCE_MAP_FOLDER):
        src = os.path.join(SOURCE_MAP_FOLDER, item)
        dest = os.path.join(DESTINATION_MAP_FOLDER, item)

        shutil.copy2(src, dest)
        copy_count += 1
        print('Complete copy %s [%d/%d]' % (item, copy_count, g_bake_count))


if __name__ == '__main__':
    print(CUTLINE)
    print('Start all')
    start_time = time.time()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    run_bake_all()
    copy_map_texture()

    consume_time = time.time() - start_time
    print('End all')
    print('Consume total time: %.4fs' % (consume_time))
    print(CUTLINE)
