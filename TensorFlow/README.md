<img src="https://www.tensorflow.org/images/tf_logo_horizontal.png" width="800px" height="200px"/>




## Installation and Environemnt Preparation

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
python -m pip install --upgrade -r requirements.txt
```



### Train

To install the latest stable version, run the following:

```shell
# Notes:

# - The `--upgrade` flag ensures you'll get the latest version.
# - The `--user` flag ensures the packages are installed to your user directory
#   rather than the system directory.
# - TensorFlow 2 packages require a pip >= 19.0
python -m pip install --upgrade --user pip
python -m pip install --upgrade --user tensorflow tensorflow_probability
```

For CPU-only usage (and a smaller install), install with `tensorflow-cpu`.

To use a pre-2.0 version of TensorFlow, run:

```shell
python -m pip install --upgrade --user "tensorflow<2" "tensorflow_probability<0.9"
```


### Test
locate youe 
To use a pre-2.0 version of TensorFlow, run:

```shell
python -m pip install --upgrade --user "tensorflow<2" "tensorflow_probability<0.9"
```

