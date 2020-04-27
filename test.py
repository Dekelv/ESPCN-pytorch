import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import cv2
import glob
import os 
import sys 
import videoPreprocessing as vp

from models import ESPCN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr

image_file = ""
video_file = ""

parser = argparse.ArgumentParser()
parser.add_argument('--weights-file', type=str, required=True)
parser.add_argument('--image-file', type=str, required=False)
parser.add_argument('--video-file', type=str, required=False)
parser.add_argument('--output-resolution', type=str, required=False)
parser.add_argument('--activation-function', type=str, required=False)
parser.add_argument('--scale', type=int, default=3)
args = parser.parse_args()

if args.video_file is not None:
    print("t1")
    vid_props = args.video_file.replace("test_videos\\","").split("_")
    print(args.video_file)
    print(vid_props)
    

name = vid_props[0]
quality = vid_props[1]
modelActivation = args.activation_function;
cwd = os.getcwd()
framesDir = "results\\frames_" + name + "_" + modelActivation + "_" + quality + "_" + str(args.scale)
highResDir = "results\\" + "superRes_" + name + "_" + modelActivation + "_" + quality + "_" + str(args.scale)
bicubicDir = "results\\" + "bicubic_" + name + "_" + modelActivation + "_" + quality + "_"+ str(args.scale)
totalPsnr = 0


if args.activation_function != 'ReLU' and args.activation_function != 'GELU':
    print(args.activation_function)
    print("unknown activation function")
    sys.exit()

def testImage(image_file):
    global totalPsnr
    image = pil_image.open(image_file).convert('RGB')
    
    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    
    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    bicubic.save(image_file.replace(framesDir, bicubicDir))

    lr, _ = preprocess(lr, device)
    hr, _ = preprocess(hr, device)
    _, ycbcr = preprocess(bicubic, device)

    print()
    with torch.no_grad():
        preds = model(lr).clamp(0.0, 1.0)

    psnr = calc_psnr(hr, preds)
    totalPsnr += psnr
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    newImageFile = image_file.replace(framesDir,highResDir).replace(".png",'').replace(".jpg",'')
    output.save(newImageFile + '{:.2f}'.format(psnr)+'.png')

def predictImage(image_file):
    global totalPsnr
    image = pil_image.open(image_file).convert('RGB')
    
    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    
    print("the image_width and height:")
    print(image_width)
    print(image_height)
    print("the image and lr")
    print(image)
    
    ## size conversion to 4k
    hr = image.resize((3840, 2160), resample=pil_image.BICUBIC)
    lr = hr.resize((image_width, image_height), resample=pil_image.BICUBIC)
    
    bicubic = lr.resize((3840, 2160), resample=pil_image.BICUBIC)
    bicubic.save(image_file.replace(framesDir, bicubicDir))

    lr, _ = preprocess(lr, device)
    hr, _ = preprocess(hr, device)
    _, ycbcr = preprocess(bicubic, device)

    print()
    with torch.no_grad():
        preds = model(lr).clamp(0.0, 1.0)

    print(hr.shape)
    print(preds.shape)
    psnr = calc_psnr(hr, preds)
    totalPsnr += psnr
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    newImageFile = image_file.replace(framesDir,highResDir).replace(".png",'').replace(".jpg",'')
    output.save(newImageFile + '{:.2f}'.format(psnr)+'.png')    


def combineImagesToBicubicVideo():
    global bicubicDir
    img_array = []
    for filename in glob.glob(bicubicDir+'/*'):
        print(filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img) 
    out = cv2.VideoWriter("results\\output_videos\\bu_" + name + "_" + modelActivation + "_" + quality + ".avi", cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def combineImagesToVideo():
    global highResDir
    img_array = []
    for filename in glob.glob(highResDir+'/*'):
        print(filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img) 
    out = cv2.VideoWriter("results\\output_videos\\SR_" + name + "_" + modelActivation + "_" + quality + ".avi", cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def createoutputDataDirectories():
    global video_file
    global framesDir
    global highResDir
    global bicubicDir

    os.mkdir(cwd + "/" + framesDir)
    os.mkdir(cwd + "/" + highResDir)
    os.mkdir(cwd + "/" + bicubicDir)


def testVideo():
    global totalPsnr
    count = 0
    for image_path in sorted(glob.glob('{}/*'.format(framesDir))):
        testImage(image_path)
        count+=1
    print("avg psnr = " + str(totalPsnr/count))

def removeFolderContent(dir):
    files = glob.glob(dir+"/*")
    for f in files:
        os.remove(f)

if __name__ == '__main__':
    if args.video_file is None and args.image_file is None:
        print("Please provide a video or image file")
        exit()
    
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = ESPCN(scale_factor=args.scale, activation=args.activation_function).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    

    #check if to run a video or image test
    if args.image_file is not None:
        image_file = args.image_file
        testImage(image_file)
    else:
        createoutputDataDirectories()
        vp.SaveFrames(args.video_file, framesDir)
        testVideo()
        combineImagesToVideo()
        combineImagesToBicubicVideo()
        #removeFolderContent('bicubic')
        #removeFolderContent('highRes')
        #removeFolderContent('tempFiles')
        print("video conversion complete")
        