#!/usr/bin/env bash

gpu=
setting=
models_folder="../models/cls/"
train_files="../data/modelnet/train_files.txt"
val_files="../data/modelnet/test_files.txt"

usage() { echo "train/val with -g gpu_id -x setting options"; }

gpu_flag=0
setting_flag=0
while getopts g:x:h opt; do
  case $opt in
  g)
    gpu_flag=1;
    gpu=$(($OPTARG))
    ;;
  x)
    setting_flag=1;
    setting=${OPTARG}
    ;;
  h)
    usage; exit;;
  esac
done

shift $((OPTIND-1))

if [ $gpu_flag -eq 0 ]
then
  echo "-g option is not presented!"
  usage; exit;
fi

if [ $setting_flag -eq 0 ]
then
  echo "-x option is not presented!"
  usage; exit;
fi

if [ ! -d "$models_folder" ]
then
  mkdir -p "$models_folder"
fi


echo "Train/Val with setting $setting on GPU $gpu!"
#CUDA_VISIBLE_DEVICES=$gpu python3 ../train_val_cls.py -t $train_files -v $val_files -s $models_folder -m pointcnn_cls -x $setting > $models_folder/pointcnn_cls_$setting.txt 2>&1 &
#CUDA_VISIBLE_DEVICES=$gpu python3 -W ignore ../train_val_cls_v2.py -t $train_files -v $val_files -s $models_folder -m pointcnn_cls_v2 -x $setting
#python3 -W ignore ../train_val_cls_v2.py -t $train_files -v $val_files -s $models_folder -m pointcnn_cls_v2 -x $setting
python3 -W ignore classify_pointcloud.py -t $train_files -v $val_files 
