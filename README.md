# cafe-picking-with-AnoGAN

Tensorflow implementation of [Anomaly GAN (AnoGAN)](https://arxiv.org/abs/1703.05921).

This model detect anomaly part in images, after training DCGAN with normal dataset.

(In Korean, H. Kim's detail explanation is [here](https://www.slideshare.net/ssuser06e0c5/anomaly-detection-with-gans))

Basic model is DCGAN (Deep Convolutional Generative Adversarial Networks).

* (Anomaly Detection of MNIST is not yet available)

## Model Description
After learn DCGAN model with normal dataset (not contains anomalies), 

* Anomaly Detector calculates anomaly score of unseen images.


![Model Structure](./assets/model_structure.jpeg)


When unseen data comes, the model tries to find latent variable z that generates input image using backpropagation. (similar with style transfer)

Anomaly Score is based on residual and discrimination losses.
- Residual loss: L1 distance between generated image by z and unseen test image.
- Discrimination loss: L1 distacne between hidden representations of generated and test image, extracted by discriminators.

![Res_Loss](./assets/res_loss.jpeg)


![Discrimination Loss](./assets/dis_loss.jpeg)

Total Loss for finding latent variable z is weighted sum of the two. (defualt lambda = 0.1)


![Total Loss](./assets/t_loss.jpeg)

## File Descriptions
- main.py : Main function of implementations, contained argument parsers, model construction, and test.
- model.py : DCGAN class (containing anomaly detection function. Imple core)
- download.py : Files for downloading celebA, LSUN, and MNIST. 
- ops.py : Some operation functions with tensorflow.
- utils.py : Some functions dealing with image preprocessing.


## Prerequisites (my environments)

- Python 2.7
- Tensorflow > 0.14
- SciPy
- pillow
- (Optional) [Align&Cropped Images.zip](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) : Large-scale CelebFaces Dataset


## Usage
まずDCGANを訓練させる必要がある。

訓練済みのファイルはcheckpointに保存する。

### Model Preparation 
(If you want to download and train the model)
First, download dataset with:

    $ python download.py mnist celebA

To train a model with downloaded dataset:

    $ python main.py --dataset mnist --input_height=28 --output_height=28 --train
    $ python main.py --dataset celebA --input_height=108 --train --crop

Or, you can use your own dataset (without central crop) by:

    $ mkdir data/DATASET_NAME
    ... add images to data/DATASET_NAME ...
    $ python main.py --dataset DATASET_NAME --train
    $ python main.py --dataset DATASET_NAME
    $ # example
    $ python main.py --dataset=eyes --input_fname_pattern="*_cropped.png" --train

### Anomaly Detection
DCGANモデルを訓練した後,test_dataフォルダーにテストデータを入れて異常検出を行う.

    $ mkdir ./test_data
    ... add test images to ./test_data ...
    
    $ python main.py --dataset DATASET_NAME --input_height=108 --crop --anomaly_test
## Dataset
コーヒー豆のデータセットは以下の Dropbox リンクにダウンロードできる。
Dropboxフォルダの中以下の4種類がある。
- OK：正常な豆のみ入る画像データ。
- NG: 異常な豆と正常な豆を混ざった画像。
- NG/label_viz:異常画像に自分でアノテーションを付けた画像。
- NG/label:アノテーションを付けた箇所のみ強調した画像。

## Results
DCGANを学習させた後、異常画像を入力した時以下の結果を出た。ここでsegmentation画像はAnoGANにより異常検出した結果である。
![result](./assets/result_example.jpeg)
## Related works
- [Image Style Transfer](https://pdfs.semanticscholar.org/7568/d13a82f7afa4be79f09c295940e48ec6db89.pdf)
- (Reconstruction-based AD) [Anomaly Detection in DBMSs](https://arxiv.org/abs/1708.02635)
- (ICLR2018 under-review) [ADGAN](https://openreview.net/forum?id=S1EfylZ0Z)

## Acknowledgement
- Thanks for @carpedm20 's implementation of [DCGAN](https://github.com/carpedm20/DCGAN-tensorflow). I implemented AnoGAN based on his implementation.
- Thanks for @LeeDoYup 's implementation of [AnoGAN](https://github.com/LeeDoYup/AnoGAN-tf).
