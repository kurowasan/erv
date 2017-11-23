#!/usr/bin/env bash
python cnn.py --n-kernel=100
python cnn.py --n-kernel=64
python cnn.py --n-kernel=32
python cnn.py --n-kernel=16
python cnn.py --n-kernel=1
python cnn.py --n-kernel=64 --kernel-dim=3,4,5,6,7
python cnn.py --n-kernel=16 --kernel-dim=3,4,5,6,7
python cnn.py --n-kernel=64 --kernel-dim=3
python cnn.py --n-kernel=16 --kernel-dim=3
python cnn.py --n-kernel=64 --lr=0.01
python cnn.py --n-kernel=64 --lr=0.001
