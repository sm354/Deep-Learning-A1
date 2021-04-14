#!/bin/sh
### Set the job name (for your reference)
#PBS -N unet
### Set the project name, your department code by default
#PBS -P ee
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=1:ngpus=1

### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=00:40:00

#PBS -l software=python
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR

module purge
module load apps/anaconda/3

python train_cifar.py --normalization torch_bn --data_dir ./ --output_file ./1.1.pth --n 1
python train_cifar.py --normalization bn --data_dir ./ --output_file ./1.2_bn.pth --n 1
python train_cifar.py --normalization in --data_dir ./ --output_file ./1.2_in.pth --n 1
python train_cifar.py --normalization bin --data_dir ./ --output_file ./1.2_bin.pth --n 1
python train_cifar.py --normalization ln --data_dir ./ --output_file ./1.2_ln.pth --n 1
python train_cifar.py --normalization gn --data_dir ./ --output_file ./1.2_gn.pth --n 1
python train_cifar.py --normalization nn --data_dir ./ --output_file ./1.2_nn.pth --n 1

#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE
