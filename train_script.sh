python train.py --train-file "91-image_x3.h5" \
                --eval-file "Set5_x3.h5" \
                --outputs-dir "outputs" \
                --scale 3 \
                --lr 1e-3 \
                --batch-size 16 \
                --num-epochs 200 \
                --num-workers 8 \
                --seed 123