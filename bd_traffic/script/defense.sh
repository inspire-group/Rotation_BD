#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
batch_size=128
for aug in 0; do
for rb_angle in 30; do
echo "Switch to cuda0"
out="./log/def_out_oneclass_clean_${aug}_${rb_angle}.txt"
echo ${out}
python other_defense.py -aug ${aug} -rb_angle ${rb_angle} -batch_size ${batch_size} > ${out}
done
done

