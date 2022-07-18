from typing import Callable, Tuple, Union, Iterable, List, Optional, Dict
import os
import json
import itertools
import shutil
import pathlib as P
from PIL import Image
import tensorflow as tf
from keras.utils import conv_utils
from tensorflow_addons.image import utils as img_utils
import numpy as np
import re
from functools import reduce




TensorLike = Union[
    tf.Tensor,
    int,
    float,
    bool,
    str,
    bytes,
    complex,
    tuple,
    list,
    np.ndarray,
    np.generic
]

class DataLoader(object):
    def __init__(   self, 
                    config: str, 
                    test_ds_dir: str, 
                    debug: bool=False
        ):
        super(DataLoader, self).__init__()

        self.noise_ = config['adding_noise'] 
        self.adding_blur_ = config['adding_blur']
        self.adding_compression_  = config['adding_compression']
        self.num_channels = config['num_channels']
        self.WH = [config['W'],config['H']]

        # configure TEST dataset
        self.debug = debug
        self.test_ds_dir = test_ds_dir
        self.test_stdVec = config['test_stdVec']
        self.test_blurVec = config['test_blurVec']
        self.test_compresVec = config['test_compresVec']
        self.img_types = config['img_types']
        self.MAX_SAMPLES = config['num_sel_imgs']

    def __call__(self):
        return self._get_test_ds()


    def _get_test_ds(self):
        test_DSs = [P.Path(sub_dir) for sub_dir in P.Path(self.test_ds_dir).glob('*/')]
        DS = {}
        for test_ds_dir in test_DSs:
            name = test_ds_dir.name
            DS[name] = self._get_test_one_ds(test_ds_dir, self.img_types, self.MAX_SAMPLES)
        return DS




    def _get_test_one_ds(self, dir_, img_types, MAX_SAMPLES): 

        # get the list of images 
        file_list_ds, ds_size = _get_image_list( dir_,
                                                img_types, 
                                                MAX_SAMPLES )

        ds_test = file_list_ds.take(ds_size)

        # take test dataset
        ds_test= ds_test.map(
            lambda x: tf.py_function(
                func=_get_image_fn,
                inp=[x, self.WH, self.num_channels], 
                Tout=(tf.float32, tf.float32, tf.string)
                )
            )

        # add noise and gibbs to the test datasets
        ds_test = self._test_noise_adder_fn( ds_test, self.noise_, self.adding_blur_,  self.adding_compression_)





        if self.debug: # True self.debug
            print( "[+] ds_test num: {}".format(len(ds_test)))

            #stdVec = self.test_stdVec
            #folder = P.Path('test_samples')
            #if not os.path.exists(folder):
            #    os.mkdir(folder)
            #utils.imwrite_Dataset(ds_test[f'y_wgn_{stdVec[-1]}'], 5, folder, self.num_channels)

        return ds_test



    # Add wgn noise to test dataset
    def _test_noise_adder_fn(self, DS, noise, 
                            adding_blur,
                            compression, 
                            num_debug = 25):
        """ Addding noise and gibbs with different stddevs and frequncies """

        if self.debug:
            return  DS.take(num_debug) 


        return {
            f'y_blur_{filter_size}x{filter_size}_compress_{compress}_wgn_{stddev}': DS.map(
                                    lambda x, y, z: tf.py_function(
                                        func=_adding_awgn_gibbs,
                                        inp=[x, y, z, stddev, filter_size, compress, noise, adding_blur, compression], 
                                        Tout=(tf.float32, tf.float32, tf.string)
                                        )
                                    ) for stddev in self.test_stdVec 
                                            for filter_size in self.test_blurVec
                                                for compress in self.test_compresVec
                }







#####################################################################
#                               Auxiliary funcs
#####################################################################

#stddev: Callable[[], int]) -> Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor] ]:



def _get_length(data_size):
  return reduce( lambda x, y: x*y, data_size ) 



def _resize_fn(img, WH):
    w, h = WH
    if h < img.get_shape()[0]:
        img = tf.image.resize(img, (h, img.get_shape()[1])) 
    elif h > img.get_shape()[0]:
        img = tf.pad(img, [[0, h-img.get_shape()[0]], [0, 0], [0, 0]])

    if w < img.get_shape()[1]:
        img = tf.image.resize(img, (img.get_shape()[0], w)) 
    elif w > img.get_shape()[1]:
        img = tf.pad(img, [[0, 0], [0, w-img.get_shape()[1]], [0, 0]])
    return img




def data_augmentation(img, opt=None):
    if opt in tf.range(1, 5):
        return tf.image.rot90(img, k=opt)
    elif opt in tf.range(5, 9):
        return tf.experimental.numpy.flipud(tf.image.rot90(img, k=opt-3))



# get image
def _get_image_fn(file_path: tf.Tensor, 
                    WH: int, 
                    channels_num: int):
    
    file = tf.io.read_file(file_path)

    # if the class of image is jpeg, use `decode_jpeg` otherwise use `decode_png`
    if tf.strings.split(tf.strings.lower(file_path), sep='.')[-1] in ['jpeg', 'jpg']:
        img = tf.io.decode_jpeg(file) 
    elif tf.strings.split(tf.strings.lower(file_path), sep='.')[-1] in ['bmp']:
        img = tf.io.decode_bmp(file)
    else:
        img = tf.image.decode_png(file, dtype=tf.uint8) 
    
    # sometimes there is error in channel conversion with decode_png or decode_jpeg
    if channels_num == 1 and img.get_shape()[-1]>1:
        img = tf.image.rgb_to_grayscale(img)

    #img = tf.cond(
    #    tf.image.is_jpeg(file),
    #    lambda: tf.image.decode_jpeg(file, channels=channels_num),
    #    lambda: tf.image.decode_png(file, channels=channels_num))


    #print(file_path, "\t", img.get_shape())

    img = tf.image.convert_image_dtype(img, tf.float32)
    #tf.print('before', "\t", img.get_shape())

    
    #img = tf.image.resize(img, WH) 
    if WH[0] > 0 and WH[1] > 0:
        img = _resize_fn(img, WH) 
    #tf.print('after', "\t", img.get_shape())
    #tf.print()


    # the image dimension must be divisible to 4
    Blocks = 16
    H_ = img.get_shape()[0] //Blocks
    W_ = img.get_shape()[1] //Blocks
    if H_ > 0 and W_ > 0:
        img = _resize_fn(img, [Blocks*W_,Blocks*H_]) 

    return img, img, file_path





def _get_image_list(data_dir, img_types, MAX_SAMPLES, shuffle=0):
    """
    ...
    """

    # get filenames
    files_pathlib = list(
        [list( data_dir.glob('**/*.'+str(img_type)) )
                for img_type in img_types]
        )

    if files_pathlib is None:
        raise ValueError("[!] The given folder is empty!")


    # flatten the lists
    files_pathlib = list(itertools.chain(*files_pathlib))
    
    #print(data_dir)
    #[print(item) for item in files_pathlib]
    #print("*"*50)

    # convert patlibs to str
    files_str = [str(fname) for fname in files_pathlib] 

    # adjust the number of images
    num_sel_imgs = len(files_str) if (MAX_SAMPLES < 1) or (MAX_SAMPLES > len(files_str)) \
                        else MAX_SAMPLES

    # the simplest way to create a dataset is to create it from a python list:
    file_list_ds = tf.data.Dataset.from_tensor_slices(files_str[:num_sel_imgs])


    # shuffle data if needed
    if shuffle > 1: 
        file_list_ds = file_list_ds.shuffle(shuffle, reshuffle_each_iteration=False)


    # get size of data
    ds_size = file_list_ds.cardinality().numpy()


    if ds_size < 1:
        raise ValueError("The images are not recognized!")

    return file_list_ds, ds_size




def _adding_awgn_gibbs( x:tf.Tensor, 
                        y:tf.Tensor, 
                        file_name:tf.string, 
                        stddev, 
                        filter_size, 
                        compress,
                        noise_, 
                        blur_,
                        compression_):
    """ adding noise and gibbs ringings to each image"""

    x_distorted = x

    if (not noise_) and (not blur_) and (not compression_):
        raise ValueError("Neither noise, nor blur, nor compression is selected!")


    #Adding blur
    if blur_:
        if isinstance(filter_size, list):
            filter_size = tf.random.uniform([], minval=filter_size[0], maxval=filter_size[-1], dtype=tf.int32)

        if filter_size%2 == 0:
            filter_size -= 1

        x_distorted =  blur_filter2d(image=x_distorted,
                                        filter_size=(filter_size,filter_size), 
                                        padding = "REFLECT", 
                                        constant_values = 0)

    #Adding compression
    if compression_:
        if tf.size(compress) == 2:
            compress = tf.random.uniform([], minval=compress[0], maxval=compress[-1])

        compress = tf.cast(int(compress), dtype=tf.int32)
        x_distorted = tf.image.random_jpeg_quality(x_distorted, compress, compress+1)



    #Adding WGN
    if noise_:
        if isinstance(stddev, list):
            stddev = tf.random.uniform([], minval=stddev[0], maxval=stddev[1])

        stddev /= 255.
                
        # For training/validation, we consider a range of stddev between [minval, maxval]
        #noise_level_map = tf.broadcast_to(stddev, tf.shape(x)[:2])[..., None]
        #wgn = tf.random.normal(tf.shape(x_tmp), stddev=stddev, name="wgn", dtype=tf.float32)
        wgn = np.random.normal(loc=0., scale=stddev, size=_get_length(tf.shape(x_distorted))) # since the other authots used np.normal, we use the same template
        x_distorted += tf.convert_to_tensor(wgn.reshape(tf.shape(x_distorted)), dtype=tf.float32)


    #Adding WGN
    if noise_:
        if isinstance(stddev, list):
            stddev = tf.random.uniform([], minval=stddev[0], maxval=stddev[1])

        stddev /= 255.
                
        # For training/validation, we consider a range of stddev between [minval, maxval]
        #noise_level_map = tf.broadcast_to(stddev, tf.shape(x)[:2])[..., None]
        #wgn = tf.random.normal(tf.shape(x_tmp), stddev=stddev, name="wgn", dtype=tf.float32)
        wgn = np.random.normal(loc=0., scale=stddev, size=_get_length(tf.shape(x_distorted))) # since the other authots used np.normal, we use the same template
        x_distorted += tf.convert_to_tensor(wgn.reshape(tf.shape(x_distorted)), dtype=tf.float32)


    return x_distorted, y, file_name







def _pad(
    image: tf.Tensor,
    filter_shape: Union[List[int], Tuple[int]],
    mode: str = "CONSTANT",
    constant_values: tf.Tensor = 0,
) -> tf.Tensor:
    """Explicitly pad a 4-D image.
    Equivalent to the implicit padding method offered in `tf.nn.conv2d` and
    `tf.nn.depthwise_conv2d`, but supports non-zero, reflect and symmetric
    padding mode. For the even-sized filter, it pads one more value to the
    right or the bottom side.
    Args:
      image: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: A `tuple`/`list` of 2 integers, specifying the height
        and width of the 2-D filter.
      mode: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC".
        The type of padding algorithm to use, which is compatible with
        `mode` argument in `tf.pad`. For more details, please refer to
        https://www.tensorflow.org/api_docs/python/tf/pad.
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode.
    """
    if mode.upper() not in {"REFLECT", "CONSTANT", "SYMMETRIC"}:
        raise ValueError(
            'padding should be one of "REFLECT", "CONSTANT", or "SYMMETRIC".'
        )
    constant_values = tf.convert_to_tensor(constant_values, image.dtype)
    filter_height, filter_width = filter_shape
    pad_top = (filter_height - 1) // 2
    pad_bottom = filter_height - 1 - pad_top
    pad_left = (filter_width - 1) // 2
    pad_right = filter_width - 1 - pad_left
    paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    return tf.pad(image, paddings, mode=mode, constant_values=constant_values)



#@tf.function
def blur_filter2d(
    image: tf.Tensor,
    filter_size: Union[int, Iterable[int]] = (3, 3),
    padding: str = "REFLECT",
    constant_values: tf.Tensor = 0,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Perform blur on image(s).
    Args:
      image: Either a 2-D `Tensor` of shape `[height, width]`,
        a 3-D `Tensor` of shape `[height, width, channels]`,
        or a 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_size: An `integer` or `tuple`/`list` of 2 integers, specifying
        the height and width of the 2-D gaussian filter. Can be a single
        integer to specify the same value for all spatial dimensions.
      padding: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC".
        The type of padding algorithm to use, which is compatible with
        `mode` argument in `tf.pad`. For more details, please refer to
        https://www.tensorflow.org/api_docs/python/tf/pad.
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode.
      name: A name for this operation (optional).
    Returns:
      2-D, 3-D or 4-D `Tensor` of the same dtype as input.
    Raises:
      ValueError: If `image` is not 2, 3 or 4-dimensional,
        if `padding` is other than "REFLECT", "CONSTANT" or "SYMMETRIC",
        if `filter_size` is invalid,
    """

    with tf.name_scope(name or "blur_filter2d"):
        filter_size = conv_utils.normalize_tuple(filter_size, 2, "filter_size")

        if any(fs < 0 for fs in filter_size):
            raise ValueError("filter_size should be greater than or equal to 1")

        image = tf.convert_to_tensor(image, name="image")

        original_ndims = img_utils.get_ndims(image)
        image = img_utils.to_4D_image(image)


        # Keep the precision if it's float;
        # otherwise, convert to float32 for computing.
        orig_dtype = image.dtype
        if not image.dtype.is_floating:
            image = tf.cast(image, tf.float32)


        channels = tf.shape(image)[3]

        filter_ = tf.ones(shape=filter_size,dtype=tf.float32, name="filter_")
        filter_ /= tf.reduce_sum(filter_)
        filter_ = filter_[:, :, tf.newaxis, tf.newaxis]
        filter_ = tf.tile(filter_, [1, 1, channels, 1])

        image = _pad(image, filter_size, mode=padding, constant_values=constant_values)

        output = tf.nn.depthwise_conv2d(
            input=image,
            filter=filter_,
            strides=(1, 1, 1, 1),
            padding="VALID",
        )
        output = img_utils.from_4D_image(output, original_ndims)
        return tf.cast(output, orig_dtype)



# Auxiliary funcs==============
def boolean_string(s):
    if s.lower()=='true':
        return True
    else:
        return False

class struct_cls:
    pass


def get_dir(_dir:Callable[[Union[list,str]],str])->list:
    if type(_dir)== str:
        _dir = list([_dir,])
    
    dirs = []
    for item in _dir:
        dirs.append("".join(list(map(lambda c: c if c not in r'[,*?!:"<>|] \\' else '', item))))
    return dirs



'''
def get_dir(_strings:Union[list,str])->list:
    get_dir_map = dict((ord(char), None) for char in r'[,*?!:"<>|]')
    dirs = []
    for _string in _strings:
        dirs.append("".join(_string.translate(get_dir_map)))
    return dirs
'''

def set_gpu_env(gpu_list:list):
    '''Set gpu/cpu mode.'''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if gpu_list:
        strategy = tf.distribute.MirroredStrategy(devices=gpu_list)
    else:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    gpus_num = strategy.num_replicas_in_sync
    

    # print the list of devices:
    tf.print(f"[i] Number of GPU devices: {gpus_num}")

    for i, gpu in enumerate(tf.config.experimental.list_physical_devices('GPU')):
        tf.print("\tDevice {}:`{}`\tType: {}".format(i, gpu.name, gpu.device_type))

    for i,cpu in enumerate(tf.config.experimental.list_physical_devices('CPU')):
        tf.print("\tlocal host {}:`{}`\tType: {}".format(i, cpu.name, cpu.device_type))


    """
    os environment:
        #0 = all messages are logged (default behavior)
        #1 = INFO messages are not printed
        #2 = INFO and WARNING messages are not printed
        #3 = INFO, WARNING, and ERROR messages are not printed
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    return strategy, gpus_num


def makedirs_fn(*argv):
    path_ = [arg for arg in argv if arg]
    path_ = os.path.join(*path_)
    if not os.path.exists(path_):
        os.makedirs(path_)
        #tf.gfile.MakeDirs(path_)
    return path_



def get_model_summary(model, model_name, short=False):
    if short:
        for layer in model.layers:
            tf.print(layer.name)
    else:
        model._name = model_name
        model.summary()

    #tf.keras.utils.plot_model(model, to_file='model_1.png', show_shapes=False)


def get_MapDataset_len(MapDataset):
    """ ... """
    # tf.data.experimental.cardinality(item).numpy())
    return MapDataset.cardinality().numpy() 


# Write some images to a folder
def imwrite_Dataset(DS, step, dst, num_channels):
    """ ... """
    if not os.path.exists(dst):
        os.makedirs(dst)

    def normalizeTo255(x):
        """ normalize data between 0 and 255 """ 
        x *= 255.
        return x.astype(np.uint8)


    counter = 0
    if isinstance(DS, dict):
        for item in DS:
            if counter%step == 0:
                if num_channels > 1:
                    Image.fromarray(normalizeTo255(item.numpy())). \
                    convert("RGB").save(os.path.join(dst, str(counter)+'.jpg'))
                else:
                    Image.fromarray(normalizeTo255(item.numpy())). \
                    save(os.path.join(dst, str(counter)+'.jpg'))

            counter += 1
    else:
        for _ in DS.take(get_MapDataset_len(DS)):
            item = next(iter(DS))
            if counter%step == 0:
                im = item[0]

                if num_channels > 1:
                    Image.fromarray(normalizeTo255(im[:,:,0:3].numpy())). \
                    convert("RGB").save(os.path.join(dst, str(counter)+'_awgn.jpg')) 

                    Image.fromarray(normalizeTo255(item[-1].numpy())). \
                    convert("RGB").save(os.path.join(dst, str(counter)+'.jpg')) 
                else:
                    Image.fromarray(normalizeTo255(im[:,:,0].numpy())). \
                    save(os.path.join(dst, str(counter)+'_awgn.jpg')) 

                    Image.fromarray(normalizeTo255(item[-1].numpy())). \
                    save(os.path.join(dst, str(counter)+'.jpg')) 
            counter += 1


# Configuration ==================
def read_config(config_dir):
    STEPS={}
    with open(config_dir, 'rb') as file:
        config = json.load(file)
    #config.get('AnyParams', anyValues???)
    
    STEPS.update({
        'val_per_step':50,
        'test_per_step':50,
        'checkpoint_per_step':50,
        'disc_per_step':3
        })

    config.update({'STEPS':STEPS})
    return config


def environ_setup(parameter, initial_val):
    if parameter in os.environ:
        return os.environ[parameter]
    else:
        return initial_val

def rmtree_dirs(dst_dir):
    if os.path.exists(dst_dir):
        try:
            shutil.rmtree(dst_dir)
            tf.print(f"'{dst_dir}' removed successfully!")
        except OSError as error:
            tf.print(error)
            tf.print(f" '{dst_dir}' can not be removed!")



def make_or_restore_model(model, step_num, ckpt_dir, options, prefix='checkpoint-'):
    """ Either restore the latest model, or load a fresh model if 
    there is no checkpoint available."""

    # check the selected #ckpt, if not exist, find the most recent one
    
    if step_num > 0:
        model_path = f"{ckpt_dir}/{prefix}{step_num}/"
        if not os.path.exists(model_path):
            step_ids = [int(dir_.split(prefix)[-1]) for dir_ in os.listdir(ckpt_dir)]
            step_num = max(step_ids)
            model_path = f"{ckpt_dir}/{prefix}{step_num}/"
    
        loaded = tf.keras.models.load_model(model_path, options=options.load_options)
        tf.print(f"[i] Restoring the model from step {step_num}.")
        return loaded

    tf.print("[i] Creating a new model.")
    return model



def restore_last_model(ckpt_dir, option_load, prefix='checkpoint-'):
    """ restore the latest model"""

    step_ids = [int(dir_.split(prefix)[-1]) for dir_ in os.listdir(ckpt_dir)]
    step_num = max(step_ids)
    model_path = f"{ckpt_dir}/{prefix}{step_num}/"
    tf.print(f"[i] Restoring the model from step {step_num}.")

    return tf.keras.models.load_model(model_path, compile=False, options=option_load)






def store_model(model_name, ckpt_dir, dst_dir, step_num, load_save_device=None):
    """ ... """

    option_load  = tf.saved_model.LoadOptions(experimental_io_device=load_save_device)
    options_save = tf.saved_model.SaveOptions(experimental_io_device=load_save_device)

    model_path = f"{ckpt_dir}/checkpoint-{step_num}/"

    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, compile=False, options=option_load)
        tf.print(f"[i] Restoring the model from step {step_num}.")

        # save as a new name in the given dir
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
    
        model.save(f'{dst_dir}/{model_name}-{step_num}', options=options_save)




# load only a model
def load_model(model_name, stored_model_dir, step_num, compile_ = False ):
    model_file = os.path.join(stored_model_dir, f'{model_name}-{str(step_num)}.h5')
    return tf.keras.models.load_model(model_file, compile=compile_)


# load checkpoints
def init_models(model, model_name, optimizer, stored_model_dir, step_num ):
    step = tf.Variable(0, dtype=tf.int64)
    checkpoint = tf.train.Checkpoint( model=model, optimizer=optimizer, step=step)
    checkpoint_file = os.path.join(stored_model_dir, f'{model_name}-{str(step_num)}.h5')
    checkpoint.restore(checkpoint_file).expect_partial()


# Extract avg and std of ALL test images
def get_overall_results(csv_filename):

    csv_file = pd.read_csv(csv_filename)  
    sgima_vals = np.unique(csv_file['sigma'])
    PSNR = {}
    SSIM = {}

    for sgima in sgima_vals:
        filt = csv_file.where(csv_file['sigma']==sgima)
        psnr = [filt['psnr'].mean(), filt['psnr'].std()]
        ssim = [filt['ssim'].mean(), filt['ssim'].std()]
        PSNR.update({sgima:psnr})
        SSIM.update({sgima:ssim})
    return PSNR, SSIM



def check_ds(ds, ds_size):

    ds_size = tf.convert_to_tensor(ds_size)
    iter_data = iter(ds)

    counter, step = tf.Variable(1., dtype=tf.float32), tf.Variable(0, dtype=ds_size.dtype)

    while  step < ds_size:
        try:
            x,y,l = next(iter_data)
            step.assign_add(1)
        except StopIteration:
            break
