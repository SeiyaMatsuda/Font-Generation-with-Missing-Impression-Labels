from PIL import Image
import glob
import os
#log_dir='./result/2020-07-29 11:03:37.049248'
log_dir='./result/2020-08-05 11:29:23.464542'
def make_randomwalk(log_dir=log_dir):
    files = sorted(glob.glob(os.path.join(log_dir,'logs_cWGAN/epoch_*.png')))
    images = list(map(lambda file : Image.open(file) , files))
    images[0].save(os.path.join(log_dir,'randomwalk.gif') , save_all = True , append_images = images , duration = 100 , loop = 0)
if "__name__"=="__main__":
    make_randomwalk()