# JCST_NIvsCG
An open source code for paper "Distinguishing Computer-Generated Images from Natural Images Using Channel and Pixel Correlation" accepted in JCST 2020.
<div align=center>
  <img width="400" src="https://github.com/Evergrow/JCST_NIvsCG/blob/master/images/img1.jpg/" hspace="10">
  <img width="400" src="https://github.com/Evergrow/JCST_NIvsCG/blob/master/images/img2.jpg/" hspace="10">
</div>

Example results of our method on images of the highly challenging dataset of [Google versus PRCG](http://www.ee.columbia.edu/ln/dvmm/downloads/PIM_PRCG_dataset/). The demo lists the confidence of two labels (NI and CG) and calculates the time cost. 

## Prerequisites
* Python 3
* NVIDIA GPU 9.0 + CUDA cuDNN 7.1.4
* PyTorch (tested on 0.4.1 and 1.0.0)
* Pillow (Python Imaging Library)
* Tqdm (Progress Bar for Python)

## Run
* Download the training and testing set: [Columbia PRCG](http://www.ee.columbia.edu/ln/dvmm/downloads/PIM_PRCG_dataset/) and [SPL2018](https://rose.ntu.edu.sg/Publications/Documents/Others/Computer%20Graphics%20Identification%20Combining%20Convolutional%20and%20Recurrent%20Neural%20Networks.pdf).
* Training: Set parameters in [train.py](https://github.com/Evergrow/JCST_NIvsCG/blob/master/train.py) or at the command line.
```python
python train.py --batch_size 64 --patch_size 96 --epoch 1200 --lr 0.001
```
* Testing: Run ```python test.py```.

## Citation
```latex
@article{zhang2020distinguishing,
  title={Distinguishing Computer-Generated Images from Natural Images Using Channel and Pixel Correlation},
  author={Zhang, Rui-Song and Quan, Wei-Ze and Fan, Lu-Bin and Hu, Li-Ming and Yan, Dong-Ming},
  journal={Journal of Computer Science and Technology},
  volume={35},
  pages={592--602},
  year={2020},
  publisher={Springer}
}
```
## Acknowledgment
This code refers from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and a Caffe version [NIvsCG](https://github.com/weizequan/NIvsCG).
