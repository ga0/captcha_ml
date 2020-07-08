import sys
import os
import random
import shutil


if __name__ == '__main__':
    src_dir, dest_dir, n = sys.argv[1:]
    n = int(n)

    src_files = os.listdir(src_dir)
    src_files = [f for f in src_files if f.endswith('.png')]

    print('src png files: %d, moved %d' % (len(src_files), n))
    chosen = random.sample(src_files, n)

    for f in chosen:
        shutil.move(src_dir + '/' + f, dest_dir)