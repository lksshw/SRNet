#!/bin/bash

#Data setup script for SRNet
#https://github.com/Niwhskal/

echo "setting up dirs ($(date))"

main_dir=$(pwd)'/'
echo "This is $main_dir"

cd "$main_dir"

mkdir "datasets/"
cd "datasets"

mkdir "srnet_data/"
mkdir -p "fonts/english_ttf/"

mkdir "bg_data"

echo "Downloading background image names ($(date))"
wget http://www.robots.ox.ac.uk/~vgg/data/scenetext/preproc/imnames.cp 

echo "----------"
echo "Downloading background images ($(date))"

cd "bg_data"
wget http://www.robots.ox.ac.uk/~vgg/data/scenetext/preproc/bg_img.tar.gz
tar -xzf bg_img.tar.gz

echo "Changing Paths $(date)"

copy_path="${main_dir}datasets/"
code_path="${main_dir}SRNet-Datagen/"

cd "${code_path}"

sed -i "s|^data_dir.*$|data_dir = '${copy_path}srnet_data/'|" cfg.py

cd "Synthtext"

sed -i "s|^font_dir.*$|font_dir = '${copy_path}fonts/english_ttf/'|" data_cfg.py

sed -i "s|^standard_font_path.*$|standard_font_path = '${copy_path}fonts/english_ttf/arial.ttf'|" data_cfg.py

sed -i "s|^bg_filepath.*$|bg_filepath = '${copy_path}imnames.cp'|" data_cfg.py

sed -i "s|^temp_bg_path.*$|temp_bg_path = '${copy_path}bg_data/bg_img/'|" data_cfg.py

echo "Moving fonts ($(date))"

cd ${main_dir}datasets/fonts/english_ttf
mv ${code_path}arial.ttf ./
mv ${code_path}OpenSans-Regular.ttf ./

cd "${code_path}"

python3 datagen.py

echo "COMPLETED AT $(date)"