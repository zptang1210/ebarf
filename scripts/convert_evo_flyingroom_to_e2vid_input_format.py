import os
import math
import h5py
import hdf5plugin
import numpy as np
import pathvalidate
import tqdm

import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils.event_utils import *



# Parameters
H_ev, W_ev = 180, 240
DATA_FOLDER = 'data/evo/flyingroom'
OUTPUT_FOLDER = 'output'
LIMIT_EVS_CONVERSIONS = None
UNDISTORT = True

# todo: temporary patch: this is the timespan used for flyingroom_ebarf_1_100.txt (overlapping)
evs_timespan_us = np.array([[  61550.67010309,  126154.5       ],
       [ 104619.83505155,  190758.16494845],
       [ 147688.83505155,  233827.16494845],
       [ 190757.83505155,  276896.66494845],
       [ 233827.33505155,  319966.16494845],
       [ 276896.83505155,  363035.16494845],
       [ 319965.83505155,  406104.16494845],
       [ 363034.83505155,  449173.16494845],
       [ 406103.83505155,  492242.66494845],
       [ 449173.33505155,  535312.16494845],
       [ 492242.83505155,  578381.16494845],
       [ 535311.83505155,  621450.16494845],
       [ 578380.83505155,  664519.16494845],
       [ 621449.83505155,  707588.66494845],
       [ 664519.33505155,  750658.16494845],
       [ 707588.83505155,  793727.16494845],
       [ 750657.83505155,  836796.16494845],
       [ 793726.83505155,  879865.16494845],
       [ 836795.83505155,  922934.66494845],
       [ 879865.33505155,  966004.16494845],
       [ 922934.83505155, 1009073.16494845],
       [ 966003.83505155, 1052142.16494845],
       [1009072.83505155, 1095211.16494845],
       [1052141.83505155, 1138280.66494845],
       [1095211.33505155, 1181350.16494845],
       [1138280.83505155, 1224419.16494845],
       [1181349.83505155, 1267488.16494845],
       [1224418.83505155, 1310557.16494845],
       [1267487.83505155, 1353626.66494845],
       [1310557.33505155, 1396696.16494845],
       [1353626.83505155, 1439765.16494845],
       [1396695.83505155, 1482834.16494845],
       [1439764.83505155, 1525903.16494845],
       [1482833.83505155, 1568972.66494845],
       [1525903.33505155, 1612042.16494845],
       [1568972.83505155, 1655111.16494845],
       [1612041.83505155, 1698180.16494845],
       [1655110.83505155, 1741249.16494845],
       [1698179.83505155, 1784318.66494845],
       [1741249.33505155, 1827388.16494845],
       [1784318.83505155, 1870457.16494845],
       [1827387.83505155, 1913526.16494845],
       [1870456.83505155, 1956595.16494845],
       [1913525.83505155, 1999664.66494845],
       [1956595.33505155, 2042734.16494845],
       [1999664.83505155, 2085803.16494845],
       [2042733.83505155, 2128872.16494845],
       [2085802.83505155, 2171941.16494845],
       [2128871.83505155, 2215010.66494845],
       [2171941.33505155, 2258080.16494845],
       [2215010.83505155, 2301149.16494845],
       [2258079.83505155, 2344218.16494845],
       [2301148.83505155, 2387287.16494845],
       [2344217.83505155, 2430356.66494845],
       [2387287.33505155, 2473426.16494845],
       [2430356.83505155, 2516495.16494845],
       [2473425.83505155, 2559564.16494845],
       [2516494.83505155, 2602633.16494845],
       [2559563.83505155, 2645702.66494845],
       [2602633.33505155, 2688772.16494845],
       [2645702.83505155, 2731841.16494845],
       [2688771.83505155, 2774910.16494845],
       [2731840.83505155, 2817979.16494845],
       [2774909.83505155, 2861048.66494845],
       [2817979.33505155, 2904118.16494845],
       [2861048.83505155, 2947187.16494845],
       [2904117.83505155, 2990256.16494845],
       [2947186.83505155, 3033325.16494845],
       [2990255.83505155, 3076394.66494845],
       [3033325.33505155, 3119464.16494845],
       [3076394.83505155, 3162533.16494845],
       [3119463.83505155, 3205602.16494845],
       [3162532.83505155, 3248671.16494845],
       [3205601.83505155, 3291740.66494845],
       [3248671.33505155, 3334810.16494845],
       [3291740.83505155, 3377879.16494845],
       [3334809.83505155, 3420948.16494845],
       [3377878.83505155, 3464017.16494845],
       [3420947.83505155, 3507086.66494845],
       [3464017.33505155, 3550156.16494845],
       [3507086.83505155, 3593225.16494845],
       [3550155.83505155, 3636294.16494845],
       [3593224.83505155, 3679363.16494845],
       [3636293.83505155, 3722432.66494845],
       [3679363.33505155, 3765502.16494845],
       [3722432.83505155, 3808571.16494845],
       [3765501.83505155, 3851640.16494845],
       [3808570.83505155, 3894709.16494845],
       [3851639.83505155, 3937778.66494845],
       [3894709.33505155, 3980848.16494845],
       [3937778.83505155, 4023917.16494845],
       [3980847.83505155, 4066986.16494845],
       [4023916.83505155, 4110055.16494845],
       [4066985.83505155, 4153124.66494845],
       [4110055.33505155, 4196194.16494845],
       [4153124.83505155, 4239263.16494845],
       [4196193.83505155, 4282332.16494845],
       [4239262.83505155, 4325401.16494845],
       [4282331.83505155, 4368470.66494845],
       [4325401.33505155, 4433074.99484536]])

if __name__ == '__main__':
    h5filename = 'events.h5'
    h5file = os.path.join(DATA_FOLDER, h5filename)
    evs_h5 = h5py.File(h5file, 'r')
    event_slicer = EventSlicer(evs_h5)

    rectify_map_filename = 'rectify_map_calib0.h5'
    rectify_map_file = os.path.join(DATA_FOLDER, rectify_map_filename)
    rmap = h5py.File(rectify_map_file, 'r')
    rectify_map = np.array(rmap['rectify_map'])
    rmap.close()

    # # * OPTION 1: pick a segment of data
    # start_time_us = np.min(evs_timespan_us)
    # end_time_us = np.max(evs_timespan_us)

    # padding_time_us = 1e6 # =1s
    # start_time_us = max(0., start_time_us - padding_time_us)
    # end_time_us = end_time_us + padding_time_us

    # ev_batch = event_slicer.get_events(start_time_us, end_time_us)
    # events = ev_batch

    # * OPTION 2: pick all data (BUG)
    start_time_us = event_slicer.get_start_time_us()
    end_time_us = event_slicer.get_final_time_us()
    num_slices = 50
    tss_slices_us = np.linspace(start_time_us, end_time_us-1, num_slices)
    ev_batches = []
    total = 0
    for i in range(len(tss_slices_us)-1):
        ev_batch = event_slicer.get_events(tss_slices_us[i], tss_slices_us[i+1])
        if ev_batch is not None:
            print(ev_batch['x'].shape)
            total += ev_batch['t'].shape[0]
            ev_batches.append(ev_batch)   
        else:
            print('skip', tss_slices_us[i], tss_slices_us[i+1])
    events = dict()
    for ev_batch_key in ev_batches[0].keys():
        events[ev_batch_key] = np.concatenate([ev_batch[ev_batch_key] for ev_batch in ev_batches])
        print(events[ev_batch_key].shape, ev_batch_key)
    
    # # print(events.shape)
    # print(total)
    # print(evs_h5['events']['t'].shape)

    evs_num = len(events["t"])
    print('total evs:', evs_num)

    # * remove distortion
    rect = rectify_map[events["y"], events["x"]]


    max_conversion_evs = evs_num if LIMIT_EVS_CONVERSIONS is None else min(evs_num, LIMIT_EVS_CONVERSIONS)
    print('#evs to write', max_conversion_evs)
    # print('t, x, y, p')

    output_file = os.path.join(OUTPUT_FOLDER, pathvalidate.sanitize_filename(DATA_FOLDER) +\
                                f'_{start_time_us}_{end_time_us}_{max_conversion_evs}_{UNDISTORT}.txt')
    with open(output_file, 'w') as fout:
        skipped = 0
        fout.write(f'{W_ev} {H_ev}\n')
        for i in tqdm.tqdm(range(max_conversion_evs)):
            if UNDISTORT == False:
                t, x, y, p = events["t"][i], events["x"][i], events["y"][i], events["p"][i]
            else:
                x_undist, y_undist = int(math.ceil(rect[i, 0])), int(math.ceil(rect[i, 1]))
                if x_undist >= 0 and x_undist < W_ev and y_undist >= 0 and y_undist < H_ev:
                    t, x, y, p = events["t"][i], x_undist, y_undist, events["p"][i] # * undistortion, but it gets a lot of float numbers and even some negative numbers, might not be usable.
                else:
                    skipped += 1
                    continue
            output_line = f'{t} {x} {y} {p}' # * x is width  y is height
            # print(output_line)
            fout.write(output_line + '\n')

    evs_h5.close()
    print('skipped:', skipped)