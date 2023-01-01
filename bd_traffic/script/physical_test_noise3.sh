#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
seed=3
for pr in 0.0005; do
for rb_angle in 15 30 45 90 ; do
for ks in $(seq 0 .01 0.3); do
out="./log/1${seed}physical_testgn.txt"
echo ${out}
echo ${seed}
python physical_test_noise.py -rb_angle ${rb_angle} -pr ${pr} -noise "gn" -var ${ks} -seed ${seed}  >> ${out}
done
done
done
