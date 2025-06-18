from pathlib import Path
from tqdm import tqdm
import imageio

src=r"/home/users/songen.gu/adwm/OccWorld/out/occworld/visgts_autoreg/200/nuScenesSceneDatasetLidar"



def create_gif(src, fps=10):
    images = []
    for img in Path(src).rglob('*.png'):
        images.append(imageio.imread(img))
    imageio.mimsave(f'{src}/vis.gif', images, fps=fps)

def create_mp4(src, fps=2):
    with imageio.get_writer(f'{src}/vis.mp4', mode='I') as writer:
        for img in Path(src).rglob('vis*.png'):
            writer.append_data(imageio.imread(img))
            
if __name__ == '__main__':
    for dir in tqdm(Path(src).iterdir()):
        if dir.is_dir():
            create_mp4(dir)
            create_gif(dir)
