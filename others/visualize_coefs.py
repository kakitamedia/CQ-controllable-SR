import argparse
import os
import yaml
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.animation import FuncAnimation, PillowWriter, ArtistAnimation

import torch

crop_size = 128

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--save_dir', default='')
    parser.add_argument('--image_id', type=int, default=0)
    parser.add_argument('--coord', nargs='+', type=int, default=(750, 265)) # 845, 265
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=256)

    return parser.parse_args()


def main():
    global args

    args = parse_args()
    assert len(args.save_dir) > 0, 'save_dir must be specified.'

    print('loading fourier predictions...')
    preds = torch.load(os.path.join(args.data_dir, f'preds_{args.image_id}.pth'))
    print('done.')

    image = Image.open(os.path.join(args.data_dir, f'num_pred_1', f'{args.image_id}.png'))
    height, width = image.size
    coord = height * args.coord[0] + args.coord[1]

    x1, y1 = args.coord[0] - crop_size//2, args.coord[1] - crop_size//2
    x2, y2 = args.coord[0] + crop_size//2, args.coord[1] + crop_size//2

    freq = torch.stack(torch.split(preds['freq'][0][0][coord], 2, dim=-1), dim=-1)
    freq_x, freq_y = freq[0], freq[1]

    freq_x = torch.stack(torch.split(freq_x, 3, dim=-1), dim=-1)
    freq_y = torch.stack(torch.split(freq_y, 3, dim=-1), dim=-1)
    freq_x_R, freq_x_G, freq_x_B = freq_x[0], freq_x[1], freq_x[2]
    freq_y_R, freq_y_G, freq_y_B = freq_y[0], freq_y[1], freq_y[2]

    coef = torch.stack(torch.split(preds['coef'][0][0][coord], 3, dim=-1), dim=-1)
    mag_R, mag_G, mag_B = coef[0], coef[1], coef[2]
    mag_R = mag_R[:mag_R.shape[-1] // 2] ** 2 + mag_R[mag_R.shape[-1] // 2:] ** 2
    mag_G = mag_G[:mag_G.shape[-1] // 2] ** 2 + mag_G[mag_G.shape[-1] // 2:] ** 2
    mag_B = mag_B[:mag_B.shape[-1] // 2] ** 2 + mag_B[mag_B.shape[-1] // 2:] ** 2

    freq_x_R, freq_y_R = freq_x_R.cpu().numpy(), freq_y_R.cpu().numpy()
    freq_x_G, freq_y_G = freq_x_G.cpu().numpy(), freq_y_G.cpu().numpy()
    freq_x_B, freq_y_B = freq_x_B.cpu().numpy(), freq_y_B.cpu().numpy()
    mag_R, mag_G, mag_B = mag_R.cpu().numpy(), mag_G.cpu().numpy(), mag_B.cpu().numpy()

    for num_pred in tqdm(range(0, freq_x_R.shape[0])):
        image = Image.open(os.path.join(args.data_dir, f'num_pred_{num_pred}', f'{args.image_id}.png'))
        image = image.crop((x1, y1, x2, y2))

        draw = ImageDraw.Draw(image)
        draw.rectangle(([(48, 48), (80, 80)]), outline='red')

        fig, axs = plt.subplots(1, 4, tight_layout=True, figsize=(20, 5))

        if num_pred > 0:
            axs[0].scatter(freq_x_R[:num_pred], freq_y_R[:num_pred], c=mag_R[:num_pred], vmin=0, vmax=max(mag_R), s=None, cmap='jet')
            axs[1].scatter(freq_x_G[:num_pred], freq_y_G[:num_pred], c=mag_G[:num_pred], vmin=0, vmax=max(mag_G), s=None, cmap='jet')
            axs[2].scatter(freq_x_B[:num_pred], freq_y_B[:num_pred], c=mag_B[:num_pred], vmin=0, vmax=max(mag_B), s=None, cmap='jet')        

        axs[0].set_xticks(np.linspace(-1.5, 1.5, 5))
        axs[0].set_yticks(np.linspace(-1.5, 1.5, 5))

        axs[1].set_xticks(np.linspace(-1.5, 1.5, 5))
        axs[1].set_yticks(np.linspace(-1.5, 1.5, 5))

        axs[2].set_xticks(np.linspace(-1.5, 1.5, 5))
        axs[2].set_yticks(np.linspace(-1.5, 1.5, 5))

        axs[3].imshow(image)

        # plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, f'num_pred_{num_pred}', f'{args.image_id}_plot.png'))
        plt.close()

    # images = []
    # for num_pred in range(0, freq_x_R.shape[0]):
    #     image = Image.open(os.path.join(args.data_dir, f'num_pred_{num_pred}', f'{args.image_id}_plot.png'))
    #     line = plt.imshow(image)
    #     images.append([line])

    # fig = plt.figure()
    # ani = ArtistAnimation(fig, images, interval=1000, blit=True)
    # ani.save(os.path.join(args.save_dir, f'{args.image_id}_plot.mp4'), writer='pillow', fps=1)

    # fig, ax = plt.subplots()
    # line = ax.imshow(images[0])

    # def update(frame):
    #     line.set_data(images[frame])
    #     return line,

    # ani = FuncAnimation(fig, update, frames=len(images), blit=True)
    # ani.save(f'{args.image_id}_plot.mp4', writer='ffmpeg', fps=1)

    # images[0].save(os.path.join(args.save_dir, f'{args.image_id}_plot.gif'), save_all=True, append_images=images[1:], duration=1000, loop=0)

if __name__ == '__main__':
    main()