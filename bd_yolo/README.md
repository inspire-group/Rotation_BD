Referred from the [official YOLOR repo](https://github.com/WongKinYiu/yolor)


This is the file for running experiments on object detection dataset

## Prepare for the experiment 

Download the [Data](http://host.robots.ox.ac.uk/pascal/VOC/) and put into `./data`.

Download the [Model](https://drive.google.com/file/d/1Tdn3yqpZ79X7R1Ql0zNlNScB1Dv9Fp76/view?usp=sharing) from [yolor](https://github.com/WongKinYiu/yolor) and put into `./checkpoints` .

## Training Backdoored Models

Object Hiding Attacks: 

```
backdoor_angle=90

rotation_augmentation=15 ##random rotation augmentation with [-15, +15] degree

inject_portion=0.01 
```

```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9532 bd_train.py --batch-size 16 --img 1280 1280 --data data/voc.yaml 
--cfg cfg/yolor_p6.cfg --weights './checkpoints/yolor_p6.pt' --device 0,1,2,3 --sync-bn --name ft001rb90df15 --hyp  hyp.finetune${rotation_augmentation}.yaml --epochs 50 --rb ${backdoor_angle} --rbrate ${inject_portion}
```

Object Misclassification Attacks: adding `--imperson` to the previous commands.  

## Evaluate Backdoored Model  
Object Hiding Attacks: 
```
python testbackdoor.py --batch-size 16 --img 1280  --data data/voc.yaml --cfg cfg/yolor_p6.cfg --weights './runs/train/ft001rb90df15/weights/best.pt' --device 0  --name att00ft001rb90df15 --verbose --rb 0 
```

"ft001rb90df15" is the name of trained backdoor model. 

Object Misclassification Attacks: using  `testbackdoor_imperson.py` to the previous commands.  