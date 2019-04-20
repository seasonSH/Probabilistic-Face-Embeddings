# Probabilistic Face Embeddings

<img src="https://raw.githubusercontent.com/seasonSH/Probabilistic-Face-Embeddings/master/assets/PFE.png" width="100%">

This is a demo code of training and testing [Probabilistic Face Embeddings]() using Tensorflow. Probabilistic Face Embeddgin (PFE) is a method that converts conventional CNN-based face embeddings into probabilistic embeddings by calibrating each feature value with an uncertainty value. The representation of each face will be an Guassian distribution parametrized by (mu, sigma), where mu is the original embedding and sigma is the learned uncertainty. Experiments show that PFE could improve the performance of face recognition by taking uncertainty into account.

## <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1000px-Tensorflow_logo.svg.png" width="25"/> Compatability
Currently this repo is compatible with Python 3 and Tensorflow r1.9.

## <img src="https://image.flaticon.com/icons/svg/149/149366.svg" width="25"/> Update Notes
+ 19/04/2019: Initial Upload.

## <img src="https://image.flaticon.com/icons/svg/182/182321.svg" width="25"/> Citation
TODO

## <img src="https://image.flaticon.com/icons/svg/1/1383.svg" width="25"/> Usage
**Note:** In this section, we assume that you are always in the directory **`$PROJECT_ROOT/`**
### Preprocessing
In this demo, we will use CASIA-WebFace, LFW and IJB-A as examples for training and testing PFEs. In this section, we will align these datasets with the landmarks I pre-extracted.
1. Download the original images of CASIA-WebFace dataset and align the images with the following command:
    ``` Shell
    python align/align_dataset.py data/ldmark_casia_mtcnncaffe.txt \
    data/casia_mtcnncaffe_aligned \
    --prefix /path/to/CASIA-Webface/images \
    --transpose_input --image_size 96 112
    ```
2. Download the original images of LFW dataset and align the images with the following command:
    ``` Shell
    python align/align_dataset.py data/ldmark_lfw_mtcnncaffe.txt \
    data/lfw_mtcnncaffe_aligned \
    --prefix /path/to/LFW/images \
    --transpose_input --image_size 96 112
    ```
3. Download the IJB-A dataset and crop the faces with the following command:
    ``` Shell
    python align/crop_ijba.py proto/IJB-A/metadata.csv \
    /path/to/IJB-A/images/ \
    data/ijba_cropped/
    ```
    To crop the images, you need to make sure that there are two folders under the given dataset folder: ```img``` and ```frame```. And then align the images with the following command:
    ``` Shell
    python align/align_dataset.py data/ldmark_ijba_mtcnncaffe.txt \
    data/ijba_mtcnncaffe_aligned \
    --prefix data/ijba_cropped \
    --transpose_input --image_size 96 112
    ```

### Training
1. Before training, you need to download the [base model](https://drive.google.com/open?id=1MiC_qCj5GFidWLtON9ekClOCJu6dPHT4). Unzip the files under ```pretrained/sphere64_caisa_am/```.

2. The configuration files for training are saved under ```config/``` folder, where you can define the training data, pre-trained model, network definition and other hyper-parameters. 
3. The uncertainty module that we are going to train is in ```models/uncertainty_module.py```.
4. Use the following command to run the default training configuration for the example base model:
    ``` Shell
    python train.py config/sphere64_casia.py
    ```
    The command will create an folder under ```log/sphere64_casia_am_PFE/```, which saves all the checkpoints and summaries. The model directory is named as the time you start training.

### Testing
+ **Single Image Comparison**
    We use LFW dataset as an example for single image comparison. Make sure you have aligned LFW images using the previous commands. Then you can test it on the LFW dataset with the following result:
    ```Shell
    python eval_lfw.py --model_dir /path/to/your/model/directory \
    --dataset_path data/lfw_mtcnncaffe_aligned
    ```

+ **Template Fusion and Comparison**
    We use IJB-A dataset as an example for template face comparison. Make sure you have aligned IJB-A images using the previous commands. Then you can test it on the LFW dataset with the following result:
    ```Shell
    python eval_ijba.py --model_dir /path/to/your/model/directory \
    --dataset_path data/ijba_mtcnncaffe_aligned
    ```
+ Note that in the original paper, I used Matlab to normalize the images, but this demo uses pure python implementation. So the performance could slightly be affected because of alignment. To use Matlab alignment code, you can check ```align/align_dataset```.

### Visualization of Uncertainty
TODO


## <img src="https://image.flaticon.com/icons/svg/48/48541.svg" width="25"/> Pre-trained Model
##### Base Model: 
[Google Drive](https://drive.google.com/open?id=1MiC_qCj5GFidWLtON9ekClOCJu6dPHT4)




