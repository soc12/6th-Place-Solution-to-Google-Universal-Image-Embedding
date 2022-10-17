Hello!

Below you can find a outline of how to reproduce my solution for the Google Universal Image Embedding competition.
If you run into any trouble with the setup/code or have any questions please contact me at socratis.gkelios@ymail.com

#ARCHIVE CONTENTS

Due to the size limit of 2GB we could not upload the pretrained model. The best submission is shared to you through our kagge notebook https://www.kaggle.com/code/socratis/guie-clip-pca

    • Code to reproduce the best submission
    • Arxiv submission pdf
    • Datasets links and instructions to process them


#HARDWARE: (The following specs were used to create the original solution)
Ubuntu 20.04.3 LTS (1 TB boot disk)
CPU : Ryzen 5900x (12 CPU Cores)
GPU : RTX 3090
RAM : 64GB (3200mhz)


#SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.9.12
cudatoolkit 11.3.1 with torch 1.11
nvidia drivers v.470.141.03

#DATA SETUP 

Data folder → Corresponding dataset [link]

cars → Stanford cars http://ai.stanford.edu/~jkrause/cars/car_dataset.html 
deepfashion →  Consumer-to-shop https://drive.google.com/drive/folders/0B7EVK8r0v71pRXllRUdQcC1zTHc?resourcekey=0-YgTkHTdQH_KN0VcXr9k_jQ
fashion200k →  Fashion200k https://www.kaggle.com/datasets/mayukh18/fashion200k-dataset
food-101 → Food-101 https://www.kaggle.com/datasets/srujanesanakarra/food101
gld → Google landmarks https://github.com/cvdfoundation/google-landmark
gldv2 → Google landmarks https://github.com/cvdfoundation/google-landmark
prod → Products10k https://products-10k.github.io/challenge.html#downloads
products → Stanford online products https://github.com/rksltnl/Deep-Metric-Learning-CVPR16
rp2k → https://www.pinlandata.com/rp2k_dataset/
ss → iMaterialist Challenge (Furniture) https://www.kaggle.com/competitions/imaterialist-challenge-furniture-2018/data
storefronts →Kaggle dataset https://www.kaggle.com/datasets/kerrit/storefront-146
in-shop →  In-shop https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E?resourcekey=0-4R4v6zl4CWhHTsUGOsTstw
met → MET artwork dataset http://cmp.felk.cvut.cz/met/
130k_kaggle → 130k public dataset from kaggle https://www.kaggle.com/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings

For gld, gldv2 and ss we provide the dataset link that have been uploaded to kaggle https://www.kaggle.com/datasets/socratis/modified-datasets

There is a train.csv which contains the paths, the original labels, the encoded labels, the set (train or valid) and the category of labels (apparel, landmark, food etc.)

The train.csv file contains all the images that we used for training. By downloading each dataset and placing them to the corresponding folder, the path should remain the same.

In case there are problems for the data placement, we have also kept the original id from each source dataset in the column ‘orig_label’ preceded by the folder’s or category’s name and the original names. For example,  the label fashion200k_90037639 refers to the original id 90037639 of the source dataset. The only exceptions are the Google Landmarks dataset, which uses the original id, and the Food-101 and Storefronts datasets, which utilize the folder’s name.

# Environment setup
using conda
conda create -n <environment-name> --file req.txt
using pip 
pip install -r requirements.txt


#MODEL BUILD (Reproduce the solution)

Train the model
    a) expect this to run 14 hours

#1) train model (overwrites models,pca,submissions in comp_dir directory)
python train.py
