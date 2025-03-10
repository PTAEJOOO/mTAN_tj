### mTAND ###
export CUDA_VISIBLE_DEVICES=0

gpu=0
seed="1 2 3 4 5"

for sd in $seed;do
    python3 src/tan_interpolation.py \
    --niters 500 --lr 0.001 --batch-size 32 \
    --rec-hidden 64 --latent-dim 16 --quantization 0.016  \
    --enc mtan_rnn --dec mtan_rnn \
    --n 8000  --gen-hidden 50 --save 1 --k-iwae 5 --std 0.01 \
    --norm --learn-emb --kl --seed $sd --num-ref-points 64 --dataset physionet --sample-tp 0.9 \
    --gpu $gpu
done
