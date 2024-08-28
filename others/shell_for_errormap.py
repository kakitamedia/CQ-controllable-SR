import os

dir = 'save/base/num_samples'

for pred_dir in os.listdir(dir):
    args = f'--gt_dir load/div2k/DIV2K_valid_HR_mini --pred_dir {dir}/{pred_dir} --save_dir {dir}/{pred_dir}/diff --scale 10.0'
    os.system(f'python others/error_map.py {args}')