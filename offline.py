import glob
import os
import pickle
from PIL import Image
from feature_extractor import FeatureExtractor

from annoy import AnnoyIndex

fe = FeatureExtractor()
a = AnnoyIndex(4096, 'angular')

img_paths = [img_path for img_path in sorted(glob.glob('static/dataset/*/*.[Jj][Pp][Gg]'))]
#png_paths = [img_path for img_path in sorted(glob.glob('static/dataset/MXR_Bookcovers/*.png'))]
#img_paths.extend(png_paths)
for idx, img_path in enumerate(img_paths):
    print(img_path)
    img = Image.open(img_path)  # PIL image
    feature = fe.extract(img)
    a.add_item(idx, feature)

a.build(-1)
print('index all images in database success!')
a.save('static/feature/dataset.tree')
pickle.dump(img_paths, open('static/feature/img_paths.pkl' ,'wb'))

