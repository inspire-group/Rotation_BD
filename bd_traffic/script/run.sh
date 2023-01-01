
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
batch_size=128
for aug in 0 15 30 45 ; do
for rb_angle in 15 30 45 90 ; do
echo "Switch to cuda0"
out="./log/out_oneclass_clean_${aug}_${rb_angle}.txt"
echo ${out}
python poison_train_oneclass_clean.py -aug ${aug} -rb_angle ${rb_angle} -batch_size ${batch_size} > ${out}
done
done

