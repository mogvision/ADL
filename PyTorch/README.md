<p align="center">
	<img src="https://upload.wikimedia.org/wikipedia/commons/c/c6/PyTorch_logo_black.svg" width="700px" height="200px"/>
</p>

This repository is the official pytorch implementation of ADL. We tested the code in the 1.11.0 environment. 

## Environment Preparation & Installation 

To install your environment (here, called ADL_env), run the following:

```shell
# Notes:
# - The `--upgrade` flag ensures you'll get the latest version.
# - The `--user` flag ensures the packages are installed to your user directory
#   rather than the system directory.
python3 -m pip install --upgrade --user pip
cd ~/
virtualenv -p python3 ADL_env
source ~/ADL_env/bin/activate
python3 -m pip install --upgrade -r requirements.txt
```



### Train

Configure ```configs/ADL_train.json``` according to your tasks:
* Denoising ->  "task_mode": "DEN"

If you use multiple gpus, add ```--distributed``` to argparse. After configuration, run the following for training RGB data:

```shell
source ~/ADL_env/bin/activate

EXPERIMENT="jsut4testRGB"
CHANNELS_NUM=3 # grey->1
python3 train.py  --DENOISER efficient_Unet \
                  --num-workers 6\
		  --EXPERIMENT ${EXPERIMENT} \
		  --json-file configs/ADL_train.json \
		  --CHANNELS-NUM ${CHANNELS_NUM} \
		  --train-dirs 'path/to/train/folder1', \
                       'path/to/train/folder2' \
		  --test-dirs 'path/to/test/folder1', \
                       'path/to/test/folder2' \
		  --distributed
```

** The code has three main steps: 1- denoiser warmup , 2- discriminator warmup, 3- ADL. You could find it as ```PHASE``` in ```train.py```.

**Models**: The denoiser and discriminator models will be stored at ```${EXPERIMENT}/ADL/checkpoints```

**Log files**: The numerical and image-bse logs will be stored at ```${EXPERIMENT}/ADL/logs```

### Test
Configure ```configs/ADL_test.json``` according to your tasks, then run the following:

```shell
EXPERIMENT="jsut4testRGB"
CHANNELS_NUM=3
python3 inference.py 	--test-dirs 'path/to/test/folder' \
          --EXPERIMENT ${EXPERIMENT} \
          --num-workers 1 \
	  --json-file configs/ADL_test.json \
	  --CHANNELS-NUM ${NUM_CHANNELS}
```



### Visualization 

The simplest way for visualization is to use tensorboard in Colab. Open a notebook at Colab. Mount your drive:
```shell
!pip install import_ipynb
from google.colab import drive
import os
drive.mount('/content/drive')
DATA_dir = '/content/drive/YourFolder'
os.chdir(Root)
```
Locate ```${EXPERIMENT}/ADL/logs``` in ```DATA_dir```, then run the following:
```shell
%load_ext tensorboard
from tensorboard import notebook
notebook.list() # View open TensorBoard instances

# Numerical results
%tensorboard --logdir "DATA_dir/numerical"

# stored images during training
%tensorboard --logdir "DATA_dir/images"
```

_______
## Citation

If you find ADL useful in your research, please cite our tech report:

```bibtex
@article{ADL2022,
    author = {Morteza Ghahremani, Mohammad Khateri, Alejandra Sierra, Jussi Tohka},
    title = {Adversarial Distortion Learning for Medical Image Denoising},
    journal = {arXiv:2204.14100},
    year = {2022},
}
```
