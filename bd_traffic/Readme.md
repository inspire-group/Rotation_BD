
This is the file for running experiments on traffic sign dataset


## Prepare for the experiment 

1. Download the [Data](https://github.com/tongwu2020/Rotation_BD/releases/download/dataset/bdt_data.zip)
2. unzip the file 
3. put them into this folder by `mv bdt_data/* ./`.


## Training Backdoored Models

Multiple Class Attacks:

```
df=SS ## other defending methods can be find in defend.py 

backdoor_angle=90

rotation_augmentation=15 ##random rotation augmentation with [-15, +15] degree

inject_portion=0.01 

python poison_train.py -aug ${rotation_augmentation} -rb_angle ${backdoor_angle} -pr ${inject_portion}
```

Command example:
```
python poison_train.py -aug 15  -rb_angle 90 -pr 0.01
```

## Defending methods 

```
python defend.py -aug ${rotation_augmentation} -rb_angle ${backdoor_angle} -defense ${df} -pr ${inject_portion}
```
Command example:
```
python defend.py -aug 15  -rb_angle 90 -defense SS -pr 0.01 
```


## Evaluate physical noise  


```
python physical_test_noise.py -rb_angle ${backdoor_angle} -pr ${inject_portion} -noise "gn"
```
Command example:
```
python physical_test_noise.py -aug 15  -rb_angle 90 -pr 0.01 -noise "gn"
```

Other noise can be found in the `./physical_test_noise.py`
