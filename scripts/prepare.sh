python prepare.py --images-dir "preprocessing/subdata" \
                --output-path "preprocessing/videoTrainData.h5" \
                --scale 3 \

python prepare.py --images-dir "preprocessing/eval" \
                --output-path "preprocessing/videoEvalData.h5" \
                --scale 3 \
                --eval \