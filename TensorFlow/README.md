<img src="https://www.tensorflow.org/images/tf_logo_horizontal.png" width="800px" height="200px"/>




## Environemnt Preparation & Installation 

To install your environment (here, called ADL_env), run the following:

```shell
# Notes:
# - The `--upgrade` flag ensures you'll get the latest version.
# - The `--user` flag ensures the packages are installed to your user directory
#   rather than the system directory.
# - TensorFlow 2 packages require a pip >= 19.0
python -m pip install --upgrade --user pip
cd ~/
virtualenv -p python3 ADL_env
source ~/ADL_env/bin/activate
python3 -m pip install --upgrade -r requirements.txt
```



### Train

Configure ```configs/ADL_train.json``` according to your tasks:
* Denoising ->  "adding_noise": true/false
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
                --train-dir 'path/to/train/folder1', \
                            'path/to/train/folder2', \
                --test-dir 'path/to/test/folder1' \
                            'path/to/test/folder2' \
                --DEBUG 'False'

```
**Data preparation**: If your directory has bad/unreadable images, remove those before the training. To do this, set --DEBUG 'True', and run the command above. It will give you the list of unreadable images. After getting the list, remove them and start training with setting --DEBUG 'False'.

**Models**: The denosier and discriminator mdoels will be stored at ```configs/ADL_train.json```

### Test
locate youe 
To use a pre-2.0 version of TensorFlow, run:

```shell
python -m pip install --upgrade --user "tensorflow<2" "tensorflow_probability<0.9"
```

### Visualization 

tesnorborad
