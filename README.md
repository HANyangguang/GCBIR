# GCBIR
Generic Content-based Image Retrieval System

## [Demo](#)
![](/demo/demoGenericCBIR.png)

## System Architecture
![](/demo/GCBIR.jpg)

## Overview
- *GCBIR* is a Generic Content-based Image Retrieval System using Keras + Flask based on deep global feature: VGG16 + Annoy. 
You can launch the search engine just by running two python scripts.
- `offline.py`: This script extracts deep features from images. Given a set of database images, a 4096D feature is extracted for each image using the VGG16 network with ImageNet pre-trained weights.
- `server.py`: This script runs a web-server. You can send your query image to the server via a Flask web-intereface. Then relevant images to the query are retrieved by the simple nearest neighbor search.
- We tested Generic-CBIR on Ubuntu 18.04 with Python3.

## Links
- [Demo](#)
- [Video](#)
- [Tutorial](https://ourcodeworld.com/articles/read/981/how-to-implement-an-image-search-engine-using-keras-tensorflow-with-python-3-in-ubuntu-18-04) and [Video](https://www.youtube.com/watch?v=Htu7b8PUyRg) by [@sdkcarlos](https://github.com/sdkcarlos)

## Usage
# Clone the code and install libraries
$ git clone 
$ cd 
$ pip install -r requirements.txt

# Put your image files (*.jpg) on static/dataset

$ python offline.py
# Then featuress of images in database are extracted and indexed by annoy, finally saved on static/feature
# Note that it takes time for the first time because Keras downloads the VGG weights.

$ export FLASK_APP=server.py
$ flask run --without-threads
# Now you can do the search via localhost:5000


## Citation

    @misc{GCBIR,
	    author = {Yangguang Han},
	    title = {GCBIR: Generic Content-based Image Retrieval System},
	    howpublished = {\url{#}}
    }

## Version note
### v0
Original source at sis: https://github.com/matsui528/sis
### v1
Add annoy(approximate nearest neighbor, oh yeah!), speed up the query time extremely. and undate the dataset incluing [Caltech 256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/), [Book Covers Dataset](https://www.kaggle.com/lukaanicin/book-covers-dataset), totally 64,251 images. 
### todo 
dimensional reducion of the 4096 featurs without losing much information.