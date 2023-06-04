#!./scripts/train_cyclegan.sh
python ../pytorch-CycleGAN-and-pix2pix/train.py --dataroot /Volumes/ExFAT/road_shoulder_gan/data/processed --name bicycle_cyclegan --model cycle_gan --gpu_ids -1