import argparse
import glob
import cv2
import numpy as np
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gt_dir', type=str, default='')
    parser.add_argument('--pred_dir', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--scale', type=float, default=10.0)

    return parser.parse_args()


def main():
    global args
    args = parse_args()

    gt_imgs = sorted(glob.glob(args.gt_dir + '/*.png'))
    pred_imgs = sorted(glob.glob(args.pred_dir + '/*.png'))

    os.makedirs(args.save_dir, exist_ok=True)

    total_error = [0, 0, 0]

    for gt, pred in tqdm(zip(gt_imgs, pred_imgs), total=len(gt_imgs)):
        gt_img = cv2.imread(gt).astype(np.float32)
        pred_img = cv2.imread(pred).astype(np.float32)

        assert gt_img.shape == pred_img.shape; 'gt and pred have different shapes'

        diff = np.abs(gt_img - pred_img)
        diff_mean = np.mean(diff, axis=0)
        diff_mean = np.mean(diff_mean, axis=1)
        for i in range(3):
            total_error[i] += diff_mean[i]
        diff = diff / args.scale
        diff = np.clip(diff, 0, 1)
        diff = (diff * 255).astype(np.uint8)

        for i, channel in enumerate(['B', 'G', 'R']):
            heatmap = cv2.applyColorMap(diff[:, :, i], cv2.COLORMAP_JET)
            save_img = heatmap * 1.0 + pred_img * 0.0
            save_path = os.path.join(args.save_dir, f'{os.path.splitext(os.path.basename(pred))[0]}_diff_{channel}_{diff_mean[i]:.2f}.png')
            cv2.imwrite(save_path, save_img)

    for i in range(3):
        total_error[i] /= len(gt_imgs)
    print(total_error)

if __name__ == '__main__':
    main()