# DSC Capstone Quarter 1: Application of Quantized Model on Subset of Original Data

## Reproduce Experiment Results
Please clone the repository and install the dependencies as follows:
1. Activate a virtual environment by
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install the requirements by
   ```
   pip3 install -r requirements.txt
   ```
3. After installing the necessary components, navigate to the `data` folder create a directory through `mkdir ILSVRC2012` and within the `ILSVRC2012` directory use the following command to download the necessary Imagenette images for experiment
   ```
   wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
   tar -xvzf imagenette2.tgz
   ```
4. You should see the imagenette2 folder, and within the folder, a train and val folders containing the training and validation sets.
5. Navigate to the `src` folder, where `main.py` is located.
6. Run the following command
   ```
   python main.py -model resnet18 -b 4 -bs 256 -s 1.16
   ```
7. Your terminal would show the quantization progress and eventually display the accuracy performance of the resnet18 model on the Imagenette data.
