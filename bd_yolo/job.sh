




python -m torch.distributed.launch --nproc_per_node 4 --master_port 9532 train.py --batch-size 16 --img 1280 1280 --data data/voc.yaml \
--cfg cfg/yolor_p6.cfg --weights './checkpoints/yolor_p6_coco.pt' --device 0,1,2,3 --sync-bn --name ft001rb90df90 --hyp  hyp.finetune90.yaml \
--epochs 50 --rb 45 --df 15 --rbrate 0.01  



python testbackdoor.py --batch-size 16 --img 1280  --data data/voc.yaml --cfg cfg/yolor_p6.cfg --weights './runs/train/ft001rb45df15/weights/best.pt' --device 0  --name att00ft001rb45df15 --verbose --rb 0 


