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
H_ev, W_ev = 720, 1280
DATA_FOLDER = 'data/TUMVIEDATA/mocap-desk2'
OUTPUT_FOLDER = 'output'
LIMIT_EVS_CONVERSIONS = None
UNDISTORT = True

# todo: temporary patch: this is the timespan used for the original config mocapDesk2_enerf.txt (no-overlapping)
evs_timespan_us = np.array([[ 3586671.56215494,  3710430.89200861],
       [ 4136809.73599139,  4710378.71050861],
       [ 5136757.55449139,  5710274.52800861],
       [ 6136653.37199139,  6710531.35100861],
       [ 7136910.19499139,  7710835.79950861],
       [ 8137214.64349139,  8710696.36600861],
       [ 9137075.20999139,  9710572.68350861],
       [10136951.52749139, 10710992.25950861],
       [11137371.10349139, 11710896.32700861],
       [12137275.17099139, 12710736.89350861],
       [13137115.73749139, 13711104.71800861],
       [14137483.56199139, 14711036.53600861],
       [15137415.37999139, 15710928.85350861],
       [16137307.69749139, 16711177.92600861],
       [17137556.76999139, 17711254.6210086 ],
       [18137633.46499139, 18711154.68850861],
       [19137533.53249139, 19711302.5100086 ],
       [20137681.35399139, 20711511.8325086 ],
       [21137890.67649139, 21261490.75384505]])

if __name__ == '__main__':
    h5filename = 'mocap-desk2-events_left.h5'
    h5file = os.path.join(DATA_FOLDER, h5filename)
    evs_h5 = h5py.File(h5file, 'r')
    event_slicer = EventSlicer(evs_h5)

    rectify_map_filename = 'rectify_map_left.h5'
    rectify_map_file = os.path.join(DATA_FOLDER, rectify_map_filename)
    rmap = h5py.File(rectify_map_file, 'r')
    rectify_map = np.array(rmap['rectify_map'])
    rmap.close()

    # * OPTION 1: pick a segment of data
    start_time_us = np.min(evs_timespan_us)
    end_time_us = np.max(evs_timespan_us)

    padding_time_us = 1e6 # =1s
    start_time_us = max(0., start_time_us - padding_time_us)
    end_time_us = end_time_us + padding_time_us

    ev_batch = event_slicer.get_events(start_time_us, end_time_us)
    events = ev_batch

    # # * OPTION 2: pick all data (BUG)
    # start_time_us = event_slicer.get_start_time_us()
    # end_time_us = event_slicer.get_final_time_us()
    # num_slices = 50
    # tss_slices_us = np.linspace(start_time_us, end_time_us-1, num_slices)
    # ev_batches = []
    # total = 0
    # for i in range(len(tss_slices_us)-1):
    #     ev_batch = event_slicer.get_events(tss_slices_us[i], tss_slices_us[i+1])
    #     if ev_batch is not None:
    #         print(ev_batch['t'].shape)
    #         total += ev_batch['t'].shape[0]
    #         # ev_batches.append(ev_batch)   
    #     else:
    #         print('skip', tss_slices_us[i], tss_slices_us[i+1])
    
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