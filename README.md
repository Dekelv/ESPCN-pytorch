# ESPCN

This repository is implementation of the ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"](https://arxiv.org/abs/1609.05158).

<center><img src="./thumbnails/fig1.png"></center>

## Requirements

- PyTorch 1.0.0
- Numpy 1.15.4
- Pillow 5.4.1
- h5py 2.8.0
- tqdm 4.30.0

## Proposal
1) Tune the Hyper-parameters
2) Possible Code Enhancements
3) Video SR
4) HR -> 4K

## Train

The 91-image, Set5 dataset converted to HDF5 using the prepare.py script

```bash
python train.py --train-file "./91-image_x3.h5" \
                --eval-file "./Set5_x3.h5" \
                --outputs-dir "./outputs" \
                --scale 3 \
                --lr 1e-3 \
                --batch-size 16 \
                --num-epochs 200 \
                --num-workers 8 \
                --seed 123                
```

## Test

The results are stored in the same path.

```bash
python test.py --weights-file "BLAH_BLAH/espcn_x3.pth" \
               --image-file "data/butterfly_GT.bmp" \
               --scale 3
```

## Results

PSNR was calculated on the **Y channel**.
Tune in for our results!

