from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import argparse
import scipy.spatial
import tqdm
import scipy.ndimage as ndimage
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
            if sigma > 15:
                sigma = 15
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.
        density += ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def generate_data(im_path):
    im = Image.open(im_path)
    im_w, im_h = im.size

    txt_path = im_path.replace('.jpg', '.txt')
    points = np.loadtxt(txt_path, delimiter=',')



    # 现在可以安全使用 points[:, 0] 和 points[:, 1]
    idx_mask = (points[:, 0] >= 0) & (points[:, 0] <= im_w) & \
            (points[:, 1] >= 0) & (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    if rr != 1.0:
        im = im.resize((im_w, im_h), Image.LANCZOS)
        points = points * rr
    return im, points


def parse_args():
    parser = argparse.ArgumentParser(description='Data processing')
    parser.add_argument('--origin-dir', default='data',
                        help='original data directory')
    parser.add_argument('--data-dir', default='processed_data',
                        help='processed data directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    min_size = 512
    max_size = 1920

    # 支持 Train / Val / Test，缺失则自动跳过
    phase_mapping = {
        'train': 'Train',
        'val': 'Val',
        'test': 'Test'
    }

    for phase, folder_name in phase_mapping.items():

        src_dir = os.path.join(args.origin_dir, folder_name)

        # 如果目录不存在则跳过
        if not os.path.isdir(src_dir):
            print(f"⚠️ {folder_name} folder not found, skip {phase}.")
            continue

        sub_dir = os.path.join(args.data_dir, phase)

        sub_dir_img = os.path.join(sub_dir, 'images')
        sub_dir_pts = os.path.join(sub_dir, 'gt_points')

        os.makedirs(sub_dir_img, exist_ok=True)
        os.makedirs(sub_dir_pts, exist_ok=True)

        # 仅 train 生成密度图
        if phase == 'train':
            sub_dir_den = os.path.join(sub_dir, 'gt_den')
            os.makedirs(sub_dir_den, exist_ok=True)

        im_list = sorted(glob(os.path.join(src_dir, "*.jpg")))

        print(f"Processing {phase} set: {len(im_list)} images")

        for im_path in tqdm.tqdm(
                im_list,
                desc=f"{phase.capitalize()} Progress"):

            im, points = generate_data(im_path)

            name = os.path.basename(im_path)

            # train生成密度图
            if phase == 'train':

                w, h = im.size

                d = np.zeros((h, w), dtype=np.float32)

                for j in range(len(points)):

                    point_x, point_y = points[j][0:2].astype(int)

                    if 0 <= point_x < w and 0 <= point_y < h:
                        d[point_y, point_x] = 1

                d = gaussian_filter_density(d)

                with h5py.File(
                        os.path.join(
                            sub_dir_den,
                            name.replace('.jpg', '.h5')),
                        'w') as hf:

                    hf['density_map'] = d

            # 保存图像
            im.save(
                os.path.join(sub_dir_img, name),
                quality=100
            )

            # 保存点标注
            np.save(
                os.path.join(
                    sub_dir_pts,
                    name.replace('.jpg', '.npy')
                ),
                points
            )

    print("✅ Data preprocessing completed.")