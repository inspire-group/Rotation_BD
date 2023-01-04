





This is the file for running experiments on face recognition dataset


## Prepare for the experiment 

Download the [Data](https://www.cs.tau.ac.il/~wolf/ytfaces/) and put into `./data` folder. Random split the data to `./data/train` , `./data/backdoor` , and `./data/test` with 8/1/1. 

Go to experiment folder to run our experiment.
```
cd Experiment 
```

Download the [VGGFace Pretrained Model](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz) (About 540M).

Unzip it and put it in this file

## Training Backdoored Models
Multiple Class Attacks:

```
df=SS ## other defending methods can be find in defend.py 

backdoor_angle=90

rotation_augmentation=1 ##random rotation augmentation with [-15, +15] degree

inject_portion=0.01 
```

```
python bd_train.py --trigger_type ${backdoor_angle} --trans ${rotation_augmentation} --inject_portion ${inject_portion} --epochs 50
```

## Defending methods 

```
python defend.py --defense ${df} --trigger_type ${backdoor_angle}  --pr ${inject_portion} 
```



