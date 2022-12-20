# GCBIR
Generic Content-based Image Retrieval System
## Interface
![](/demo/demoGenericCBIR.png)

## Architecture
![](/demo/GCBIR.jpg)

## Overview
- *GCBIR* is a Generic Content-based Image Retrieval System using Keras + Flask based on CNN deep global feature(VGG16) and  approximate nearest neighbor search method(Annoy). 
You can launch the system just by running two python scripts.
- `offline.py`: This script extracts deep features from images. Given a set of database images, a 4096D feature is extracted for each image using the VGG16 network with ImageNet pre-trained weights.
- `server.py`: This script runs a web-server. You can send your query image to the server via a Flask web-intereface. Then relevant images to the query are retrieved by approximate nearest neighbor search Annoy.
- We tested Generic-CBIR on Ubuntu 18.04 with Python3.

## Usage
Clone the code and install libraries
```bash
$ git clone https://github.com/HANyangguang/GCBIR.git
$ cd GCBIR
$ pip(3) install -r requirements.txt
```

Put your image files (*.jpg) on static/dataset

```bash
$ python offline.py
```

Update dirs
```bash
$ mkdir static/feature
$ mkdir static/uploaded
```

Then featuress of images in database are extracted and indexed by annoy, finally saved on static/feature. Note that it takes time for the first time because Keras downloads the VGG weights.
```bash
$ export FLASK_APP=server.py
$ flask run --without-threads
```
Now you can do the search via localhost:5000

## Acknowledgment
[ANNOY](https://github.com/spotify/annoy)     
[VGG16](https://arxiv.org/abs/1409.1556)    
[Flask](https://github.com/matsui528/sis)      
[Tutorial](https://ourcodeworld.com/articles/read/981/how-to-implement-an-image-search-engine-using-keras-tensorflow-with-python-3-in-ubuntu-18-04) and [Video](https://www.youtube.com/watch?v=Htu7b8PUyRg) by [@sdkcarlos](https://github.com/sdkcarlos)
