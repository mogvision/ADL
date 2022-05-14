<p align="center">
	<img src="https://www.tensorflow.org/images/tf_logo_horizontal.png" width="700px" height="200px"/>
</p>

This repository is the official TensorFlow implementation of ADL. We tested the code in the TF2.5.0 environment. 

## Environment Preparation & Installation 

To install your environment (here, called ADL_env), run the following:

```shell
# Notes:
# - The `--upgrade` flag ensures you'll get the latest version.
# - The `--user` flag ensures the packages are installed to your user directory
#   rather than the system directory.
# - TensorFlow 2 packages require a pip >= 19.0
python3 -m pip install --upgrade --user pip
cd ~/
virtualenv -p python3 ADL_env
source ~/ADL_env/bin/activate
python3 -m pip install --upgrade -r requirements.txt
```



### Train

Configure ```configs/ADL_train.json``` according to your tasks:
* Denoising ->  "adding_noise": true/false
* Deblurring -> "adding_blur": true/false
* Compression -> "compression": true/false

After configuration, run the following for training RGB data:

```shell
source ~/ADL_env/bin/activate

EXPERIMENT="testRGB"
CHANNELS_NUM=3 # grey->1
python3 train.py --DENOISER Efficient_Unet \
                --EXPERIMENT ${EXPERIMENT} \
                --json-file configs/ADL_train.json \
                --channels-num ${CHANNELS_NUM} \
                --train-dirs 'path/to/train/folder1', \
                            'path/to/train/folder2' \
                --test-dirs 'path/to/test/folder1', \
                            'path/to/test/folder2' \
                --DEBUG 'False'

```
**Data preparation**: If your directory has bad/unreadable images, remove those before the training. To do this, set --DEBUG 'True', and run the command above. It will give you a list of unreadable images. After getting the list, remove them and start training by setting --DEBUG 'False'.

**Models**: The denoiser and discriminator models will be stored at ```${EXPERIMENT}/ADL/checkpoints```

**logs**: The logs will be stored at ```${EXPERIMENT}/ADL/logs```.


### Test
Configure ```configs/ADL_test.json``` according to your tasks, then run the following:

```shell
EXPERIMENT="testRGB"
CHANNELS_NUM=3
python3 inference.py --test-dirs 'path/to/test/folder'  \
		--num-channels ${CHANNELS} \
		--EXPERIMENT ${EXPERIMENT} \
		--json-file configs/ADL_test.json
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
