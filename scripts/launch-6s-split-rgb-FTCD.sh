#!/usr/bin/env bash
# return to the root directory

CUDA=0
PORT=1234
NAME=rgb-ftcd
DATA_ROOT="..."
COMMENT="Retraining with RGB, finetuning with unbiased CE + contrastive distillation"

# back to main folder
cd ..
for STEP in {1..4}
do
    echo "===| Launching step ${STEP}... |==="

    CUDA_VISIBLE_DEVICES=$CUDA accelerate launch --config configs/single-gpu.json --main_process_port $PORT run.py train \
    --data-root $DATA_ROOT \
    --method=UFT \
    --model.encoder=tresnet_m \
    --task.name 6s \
    --task.step $STEP \
    --task.filter-mode=split \
    --model.act=ident \
    --model.norm=iabn_sync \
    --trainer.batch-size=8 \
    --trainer.num-workers=4 \
    --trainer.amp \
    --trainer.patience=25 \
    --optimizer.lr=1e-3 \
    --scheduler.target=cosine \
    --in-channels=3 \
    --aug.factor=0.1 \
    --aug.factor-icl=10.0 \
    --name=$NAME \
    --comment $COMMENT
done
