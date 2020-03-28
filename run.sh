#!/bin/bash
#block(name=block-1, threads=5, memory=2000, subtasks=1, gpus=1, hours=24)
python3 main.py --data_dir "../data/h3.6m/dataset/" --epoch 10 --input_n 50 --output_n 50 --dct_n 35 --exp "logs/"
