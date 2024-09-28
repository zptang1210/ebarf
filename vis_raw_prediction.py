import os
import shutil
import glob
import numpy as np
import cv2
from tqdm import tqdm

FULL_FOLDER_PATH = '/home/zhipengtang_umass_edu/erik/nerf/enerf/output/eds_output/sep-16-ebarf-eds00-consecutive-overlap-shorter-aug_eds00_ebarf_configs_eds00_eds00_ebarf_consecutive_overlap_shorter_aug'
INPUT_DIR = os.path.join(FULL_FOLDER_PATH, 'validation', 'raw')
NUM_VIS = 100
OUTPUT_DIR = './output/tmp_vis_output'
mode = 'eds' # tumvie/eds/a b

all_files = glob.glob(os.path.join(INPUT_DIR, '*'))
all_files = all_files[-NUM_VIS:]

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_preds = []
for fpath in all_files:
       pred = np.load(fpath, allow_pickle=True)
       all_preds.append(pred)

preds_logs = [np.log(255*im + 1e-3) for im in all_preds]
preds_logs = np.stack(preds_logs) # (len(val_idxs), H, W, C)

# imgs_gt_log = [torch.log(255*im[0] + 1e-3) for im in all_gts]
# imgs_gt_log = torch.stack(imgs_gt_log)

# a, b = solve_normal_equations(preds_logs, imgs_gt_log) #todo:this method is problematic in eds00
# print('computed ab:', a, b)
if mode == 'eds': #todo
    a, b = 1.6677236175621883, -3.5166376893621627
elif mode == 'tumvie':
    a, b = 3.085902418441004, -10.835323226113228
else:
    try:
        a, b = tuple(map(float, mode.split(' ')))
    except:
        raise ValueError
print('reset ab to', a, b)

for j in tqdm(range(len(preds_logs))):
    pred_cor_j = np.exp(preds_logs[j] * a + b)
    # saving rgb-view prediction. clip image to range and draw text
    pred_corrected_img = np.clip(pred_cor_j, a_min=0, a_max=255)
    pred_corrected_img = np.rint(pred_corrected_img).astype(np.uint8)

    fname = os.path.splitext(os.path.basename(all_files[j]))[0]
    save_path_pred = os.path.join(OUTPUT_DIR, f'{fname}.png')
    cv2.imwrite(save_path_pred, pred_corrected_img)