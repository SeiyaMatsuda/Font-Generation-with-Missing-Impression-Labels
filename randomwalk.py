from PIL import Image
import glob
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=str)
opts = parser.parse_args()
def make_randomwalk(log_dir):
    path = os.path.join(log_dir, '*.png')
    files = sorted(glob.glob(path))
    images = list(map(lambda file : Image.open(file) , files))[::3]
    images = [i.resize((i.size[0]//4,i.size[1]//4)) for i in images]
    images[0].save(os.path.join(log_dir, 'randomwalk.gif'), save_all=True, append_images=images, duration=15, loop=0)

if __name__ == '__main__':
    make_randomwalk(opts.logdir)