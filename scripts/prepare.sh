python prepare.py --images-dir "preprocessing/train" --output-path "preprocessing/videoTrainData.h5" --scale 4

python prepare.py --images-dir "eval" --output-path "preprocessing/videoEvalData.h5" --scale 2 --eval 