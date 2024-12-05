## CNN Quantization & Evaluation on ImageNet Subset

Authors:
- Saathvik Dirisala
- Jessica Hung
- Ari Juljulian

This repository is contains various experiments we conducted over the course of the quarter as a part of DSC 180A. Our overarching research question has been: How is quantizibiltiy of CNNs impacted by the distribution of the calibration data?

In order to answer this question, we mainly focused our experimentation on the GPFQ algorithm, which reported impressively low loss in accuracy even in 3-bit precision. We also experimented with toy models and simple quantization techniques to understand the algorithmic foundations of quantization. Altogether, our experimentation has allowed us to answer some important questions related to our project and also lay out a path for future research.

## Quantization Experiments

### Create and activate python environment
```
conda create --name myenv python=3.11.10
conda activate myenv
```

### Clone repository
```
git clone https://github.com/saathvikpd/DSC180AB_Capstone.git
```

### Change directory and install requirements
```
cd ./DSC180AB_Capstone/
pip3 install -r requirements.txt
```

### Start Jupyter Server
```
jupyter notebook
```

Open Experimentation.ipynb, run all the cells in the notebook, and observe results


## Replicating GPFQ algorithm for a subset of ImageNet dataset

Move up one directory level, out of "DSC180AB_Capstone", if necessary.

### Clone relevant repo
```
git clone https://github.com/YixuanSeanZhou/Quantized_Neural_Nets.git
```

### Instead of downloading ImageNet, download ImageWoof from:

https://github.com/fastai/imagenette?tab=readme-ov-file

- Download the 160px dataset of dog breeds
- Move the folders relevant to training data into data/ILSVRC2012/ILSVRC2012_img_train/
- Move the folders relevant to validation data into data/ILSVRC2012/ILSVRC2012_img_val/

### Replace the files from the original repo with the modified files under ./GPFQ_modified_files

```
rm ./Quantized_Neural_Nets/src/data_loaders.py
rm ./Quantized_Neural_Nets/src/main.py
rm ./Quantized_Neural_Nets/src/utils.py
rm ./Quantized_Neural_Nets/data/ILSVRC2012/wnid_to_label.pickle

mv ./GPFQ_modified_files/data_loaders.py ./Quantized_Neural_Nets/src/
mv ./GPFQ_modified_files/main.py ./Quantized_Neural_Nets/src/
mv ./GPFQ_modified_files/utils.py ./Quantized_Neural_Nets/src/
mv ./GPFQ_modified_files/wnid_to_label.pickle ./Quantized_Neural_Nets/data/ILSVRC2012/
mv ./GPFQ_modified_files/wnid_to_label_org.pickle ./Quantized_Neural_Nets/data/ILSVRC2012/
```

Follow the instructions in ./Quantized_Neural_Nets/README.md to run the code for GPFQ


