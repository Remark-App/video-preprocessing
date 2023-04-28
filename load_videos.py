import numpy as np
import pandas as pd
import imageio
import os
import subprocess
from multiprocessing import Pool
from itertools import cycle
import warnings
import glob
import time
from tqdm import tqdm
from util import save
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
warnings.filterwarnings("ignore")

DEVNULL = open(os.devnull, 'wb')

def download(video_id, args):
    video_path = os.path.join(args.video_folder, video_id + ".mp4")
    subprocess.call([args.youtube, '-f', "''best/mp4''", '--write-auto-sub', '--write-sub',
                     '--sub-lang', 'en', '--skip-unavailable-fragments',
                     "https://www.youtube.com/watch?v=" + video_id, "--output",
                     video_path], stdout=DEVNULL, stderr=DEVNULL)
    return video_path


def make_square(x1, y1, x2, y2, max_x, max_y, scale=1.0):
    # 计算矩形的宽度和高度
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    # 找到最大的边长
    max_side = max(width, height)

    # 计算矩形中心点
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # 计算新的左上角和右下角坐标
    new_x1 = int(center_x - max_side / 2 * scale)
    new_y1 = int(center_y - max_side / 2 * scale)
    new_x2 = int(center_x + max_side / 2 * scale)
    new_y2 = int(center_y + max_side / 2 * scale)

    # 检查是否超出画布范围
    if new_x1 < 0:
        offset = -new_x1
        new_x1 += offset
        new_x2 += offset
    elif new_x2 > max_x:
        offset = new_x2 - max_x
        new_x1 -= offset
        new_x2 -= offset

    if new_y1 < 0:
        offset = -new_y1
        new_y1 += offset
        new_y2 += offset
    elif new_y2 > max_y:
        offset = new_y2 - max_y
        new_y1 -= offset
        new_y2 -= offset

    return new_x1, new_y1, new_x2, new_y2

def run(data):
    video_id, args = data
    if not os.path.exists(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')):
       download(video_id.split('#')[0], args)

    if not os.path.exists(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')):
       print ('Can not load video %s, broken link' % video_id.split('#')[0])
       return
    try:
        reader = imageio.get_reader(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4'))
    except:
        download(video_id.split('#')[0], args)
        try:
            reader = imageio.get_reader(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4'))
        except:
            print ('Can not load video %s, broken link' % video_id.split('#')[0])
            return

    fps = reader.get_meta_data()['fps']

    df = pd.read_csv(args.metadata)
    df = df[df['video_id'] == video_id]
    
    all_chunks_dict = [{'start': df['start'].iloc[j], 'end': df['end'].iloc[j],
                        'bbox': list(map(int, df['bbox'].iloc[j].split('-'))), 'frames':[]} for j in range(df.shape[0])]
    ref_fps = df['fps'].iloc[0]
    ref_height = df['height'].iloc[0]
    ref_width = df['width'].iloc[0]
    partition = df['partition'].iloc[0]
    try:
        for i, frame in enumerate(reader):
            for entry in all_chunks_dict:
                if 'person_id' in df:
                    first_part = df['person_id'].iloc[0] + "#"
                else:
                    first_part = ""
                first_part = first_part + '#'.join(video_id.split('#')[::-1])
                path = first_part + '#' + str(entry['start']).zfill(6) + '#' + str(entry['end']).zfill(6) + '.mp4'
                # if exists, do not save
                if os.path.exists(os.path.join(args.out_folder, partition, path)):
                    continue
                if (i * ref_fps >= entry['start'] * fps) and (i * ref_fps < entry['end'] * fps):
                    left, top, right, bot = entry['bbox']
                    left = int(left / (ref_width / frame.shape[1]))
                    top = int(top / (ref_height / frame.shape[0]))
                    right = int(right / (ref_width / frame.shape[1]))
                    bot = int(bot / (ref_height / frame.shape[0]))
                    left, top, right, bot = make_square(left, top, right, bot, frame.shape[1], frame.shape[0], 1.1)
                    crop = frame[top:bot, left:right]
                    if args.image_shape is not None:
                       crop = img_as_ubyte(resize(crop, args.image_shape, anti_aliasing=True))
                    entry['frames'].append(crop)
    except Exception:
        pass
    try:
        for entry in all_chunks_dict:
            if not entry['frames']:
                continue
            if 'person_id' in df:
                first_part = df['person_id'].iloc[0] + "#"
            else:
                first_part = ""
            first_part = first_part + '#'.join(video_id.split('#')[::-1])
            path = first_part + '#' + str(entry['start']).zfill(6) + '#' + str(entry['end']).zfill(6) + '.mp4'
            save(os.path.join(args.out_folder, partition, path), entry['frames'], args.format)
            print('Done %s' % path)
    except:
        pass
    print('Done %s' % video_id)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video_folder", default='youtube-taichi', help='Path to youtube videos')
    parser.add_argument("--metadata", default='taichi-metadata-new.csv', help='Path to metadata')
    parser.add_argument("--out_folder", default='taichi-png', help='Path to output')
    parser.add_argument("--format", default='.png', help='Storing format')
    parser.add_argument("--workers", default=1, type=int, help='Number of workers')
    parser.add_argument("--youtube", default='./youtube-dl', help='Path to youtube-dl')
 
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape, None for no resize")

    args = parser.parse_args()
    if not os.path.exists(args.video_folder):
        os.makedirs(args.video_folder)
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    for partition in ['test', 'train']:
        if not os.path.exists(os.path.join(args.out_folder, partition)):
            os.makedirs(os.path.join(args.out_folder, partition))

    df = pd.read_csv(args.metadata)
    video_ids = set(df['video_id'])
    pool = Pool(processes=args.workers)
    args_list = cycle([args])
    for chunks_data in tqdm(pool.imap_unordered(run, zip(video_ids, args_list))):
        pass
