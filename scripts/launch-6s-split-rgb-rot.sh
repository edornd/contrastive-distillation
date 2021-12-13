#!/usr/bin/env bash
# return to the root directory

CUDA=0
PORT=1234
NAME=rgb-mib-cd
DATA_ROOT="..."
COMMENT="Retraining with RGB, rot. invariance on both new and old, factor 0.1, flip+rot90"

# back to main folder
cd ..
for STEP in {0..4}
do
    echo "===| Launching step ${STEP}... |==="

    CUDA_VISIBLE_DEVICES=$CUDA accelerate launch --config configs/single-gpu.json --main_process_port $PORT run.py train \
    --data-root $DATA_ROOT \
    --model.encoder=tresnet_m \
    --task.name 6s \
    --task.step $STEP \
    --task.filter-mode=split \
    --model.act=ident \
    --model.norm=iabn_sync \
    --trainer.batch-size=8 \
    --trainer.amp \
    --trainer.patience=25 \
    --optimizer.lr=1e-3 \
    --scheduler.target=cosine \
    --in-channels=3 \
    --aug.factor=0.1 \
    --aug.factor-icl=0.1 \
    --aug.fixed-angles \
    --name=$NAME \
    --comment $COMMENT
done
