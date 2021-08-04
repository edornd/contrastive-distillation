#!/usr/bin/env bash

export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --logdir outputs --port 6006