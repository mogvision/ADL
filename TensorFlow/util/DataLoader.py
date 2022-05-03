from typing import Callable, Tuple, Dict, Optional, Union, List, Iterable
import os
import pathlib
from tensorflow_addons.image import utils as img_utils
from keras.utils import conv_utils
import tensorflow as tf
import numpy  as np
from functools import reduce
from operator import concat

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



from util import utils

LABEL_noisy_img = tf.constant(0., dtype=tf.float32)
LABEL_true_img  = tf.constant(1., dtype=tf.float32)


class DataLoader(object):
    def __init__(self,
                strategy,
                gpus_num:int,
                channels_num:int,
                config: str,
                train_ds_dir: str,
                test_ds_dir: str, 
                debug: bool
        )->Dict[str, tf.data.Dataset]:
        
        super(DataLoader, self).__init__()
        self.strategy = strategy
        self.gpus_num=gpus_num
        self.debug = debug

        self.conf_train = config['train_ds']
        self.conf_test  = config['test_ds']

        # datasets specification
        self.shuffle = config['shuffle']
        self.bs = config['batch_size_per_gpu']

        self.WH = [config['W'], config['H']]
        self.channels_num = channels_num

        # configure TRAIN
        self.train_ds_dir = train_ds_dir
        self.gt_noisy=config['gt_noisy']
        self.adding_noise = config['adding_noise']
        self.adding_blur = config['adding_blur']
        self.compression  = config['compression']
        self.train_std_interval   = config["train_std_interval"]
        self.train_blur_interval = config["train_blur_interval"]
        self.train_compres_interval = config["train_compres_interval"] 
        self.validation_std_interval= config["validation_std_interval"]

        # configure TEST 
        self.test_ds_dir = test_ds_dir
        if self.adding_noise:
            self.test_stdVec = config["test_stdVec"]
        else:
            self.test_stdVec = [0.0]

        if self.adding_blur:
            self.test_blurVec = config["test_blurVec"]
        else:
            self.test_blurVec = [0.0]

        if self.compression:
            self.test_compresVec = config["test_compresVec"]
        else:
            self.test_compresVec = [0.0]



    def __call__(self):
        tf.print("[i] Loading data...")

        try:
            # Get image list
            ds_train, ds_val, num_train, num_val = self.get_train_val_ds()
            ds_test, num_test = self.get_test_ds()

            # Mirror data by distribute_datasets (xgpus)
            Mirrored_ds_train = self.strategy.distribute_datasets_from_function(
                lambda _: self.prepare_ds(ds_train, 'training'))

            Mirrored_ds_val = self.strategy.distribute_datasets_from_function(
                lambda _: self.prepare_ds(ds_val, 'validation'))


            Mirrored_ds_test = self.strategy.distribute_datasets_from_function(
                lambda _: self.prepare_test_ds(ds_test))

            # Calculate the batch size after applying distribute_datasets
            steps_per_epoch = {}

            steps_per_epoch.update({'TRAIN':num_train//self.bs})
            steps_per_epoch.update({'VAL':num_val//self.bs})
            steps_per_epoch.update({'TEST':num_test//self.bs})

            tf.print(f"\t[+] Train includes {num_train} images with {steps_per_epoch['TRAIN']} batches for being processed by {self.gpus_num} gpus.")
            tf.print(f"\t[+] Validation includes {num_val} images with {steps_per_epoch['VAL']} batches for being processed by {self.gpus_num} gpus.")
            tf.print(f"\t[+] Test includes {num_test} images with {steps_per_epoch['TEST']} batches for being processed by {self.gpus_num} gpus.")
            
            if self.gpus_num>1:
                tf.print(f'\tWarning! drop_remainder for training/validation is `True`!')
            return Mirrored_ds_train, Mirrored_ds_val, Mirrored_ds_test, steps_per_epoch

        except MemoryError:
            raise ValueError("Warning! There is not enough memory for loading data!")




    # ************************************
    # Auxiliary funcs 
    # ************************************
    def get_train_val_ds(self):
        MAX_VAL_IMGS = self.conf_train['num_val_max']

        file_list_ds, ds_size = self._get_images( self.train_ds_dir, 
                                    self.conf_train['img_types'], 
                                    self.conf_train['num_sel_imgs'],
                                    self.shuffle)

        num_val = int( (1.-self.conf_train['train_val_ratio']) * ds_size)
        num_val = MAX_VAL_IMGS if num_val > MAX_VAL_IMGS else num_val

        num_train = ds_size - num_val
        ds_train = file_list_ds.take(num_train)
        ds_val   = file_list_ds.skip(num_train)
        return ds_train, ds_val, num_train, num_val


    def prepare_ds(self, ds, status):
        # take train dataset
        ds = ds.map( lambda x: tf.py_function(
                        func=_get_paired_image_fn,
                        inp=[x, self.WH, self.channels_num, self.gt_noisy, "Training_Valid", self.debug], 
                        Tout=(tf.float32, tf.float32)
                        ),
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False)

        # add noise to the train dataset
        DS = self._train_distortion_adder_fn(ds, 
                                    self.gt_noisy,
                                    self.adding_noise, 
                                    self.adding_blur, 
                                    self.compression,
                                    status)

        # create batch files
        DS = DS.batch(self.bs, drop_remainder=True) 

        return DS



    def get_test_ds(self): 
        # get images
        file_list_ds, ds_size = self._get_images( self.test_ds_dir, 
                                    self.conf_test['img_types'], 
                                    self.conf_test['num_sel_imgs']
                                    )
        ds_test = file_list_ds.take(ds_size)
        return ds_test, ds_size



    def prepare_test_ds(self, ds):
        # take test dataset
        ds= ds.map(
                    lambda x: tf.py_function(
                        func=_get_paired_image_fn,
                        inp=[x, self.WH, self.channels_num, self.gt_noisy, 'TEST', self.debug], 
                        Tout=(tf.float32, tf.float32)
                        ),
                        num_parallel_calls=tf.data.AUTOTUNE,
                        deterministic=False
                    )

        DS = self._train_distortion_adder_fn( ds, 
                                    self.gt_noisy,
                                    self.adding_noise, 
                                    self.adding_blur,
                                    self.compression, 
                                    status='test_random')

        
        # create batch files. This is just for evaluting test set in the dimension of TRAIN set.
        DS = DS.batch(self.bs, drop_remainder=True) 

        return DS



    def _get_images(self, data_dirs, img_types, MAX_SAMPLES, shuffle=0):
        """
        ...
        """
        #[print( dir_ , os.path.exists(dir_), "*"*25) for dir_ in utils.get_dir(data_dirs) ]

        # get files
        FILES = [os.path.join(path_, name_) 
                    for dir_ in utils.get_dir(data_dirs)
                        for path_, subdirs, files_ in os.walk(dir_)
                            for name_ in files_ 
                                if name_.lower().endswith(tuple(img_types))
                ]
        #tf.print(FILES, ' <'*20)

        if FILES is None:
            raise NameError('[!] `{}` is empty!'.format(data_dir))

        # adjust the number of images
        num_sel_imgs = len(FILES) if (MAX_SAMPLES < 1) or (MAX_SAMPLES > len(FILES)) \
                        else MAX_SAMPLES

        # the simplest way to create a dataset is to create it from a python list:
        file_list_ds = tf.data.Dataset.from_tensor_slices(FILES[:num_sel_imgs])


        # shuffle data if needed
        if shuffle > 1: 
            file_list_ds = file_list_ds.shuffle(shuffle, reshuffle_each_iteration=False)


        # get size of data
        ds_size = file_list_ds.cardinality().numpy()


        if ds_size < 1:
            raise ValueError("The images are not recognized!")

        return file_list_ds, ds_size




    # Add wgn noise to train dataset
    def _train_distortion_adder_fn(self, DS, 
                                    gt_noisy:bool,
                                    adding_noise:bool, 
                                    adding_blur:bool, 
                                    adding_compression:bool,  
                                    status: str=None):

        """Addding distortion with a random defined in the config
        Args:
            adding_noise: adding WGN or not
            adding_blur: adding blur or not
            adding_compression: Compress image or not

        Returns:
            DS
        """

        def train_adding_distortion_fn(x:tf.Tensor, gt:tf.Tensor,
                                        gt_noisy:bool,
                                        adding_noise:bool, 
                                        adding_blur:bool,
                                        adding_compression:bool,
                                        stddevVec:Tuple[float,float], 
                                        filter_size:Tuple[int,Iterable[int]],
                                        compressVec:int
                                        ):
            """ adding noise to each image"""
            x_distorted = x

            if (not gt_noisy) and (not adding_noise) and (not adding_blur) and (not adding_compression):
                raise ValueError("Neither gt_noise, nor noise, nor blur, nor compression is selected!")


            """
            Adding Blur 
            """
            if adding_blur:
                if tf.size(filter_size) == 2:
                    filter_size = tf.random.uniform([], minval=filter_size[0], maxval=filter_size[-1], dtype=tf.int32)

                if filter_size%2 == 0:
                    filter_size -= 1

                x_distorted =  blur_filter2d(image=x_distorted,
                                            filter_size=(filter_size,filter_size), 
                                            padding = "REFLECT", 
                                            constant_values = 0)

            """
            Jpeg compression
            compressVec: Define the fraction of Jpeg compression
            """
            if adding_compression:
                # For training/validation, we consider a range of stddev between [minval, maxval]
                if tf.size(compressVec) == 2:
                    compressVec = tf.random.uniform([], minval=compressVec[0], maxval=compressVec[-1])

                compressVec = tf.cast(int(compressVec), dtype=tf.int32)

                x_distorted = tf.image.random_jpeg_quality(x_distorted, compressVec, compressVec+1)
                #label = tf.constant(LABEL_true_img) if compressVec <1 else tf.constant(LABEL_noisy_img)


            """
            Adding WGN
            """
            if adding_noise:
                # For training/validation, we consider a range of stddev between [minval, maxval]
                if tf.size(stddevVec) == 2:
                    stddev = tf.random.uniform([], minval=stddevVec[0], maxval=stddevVec[-1])
                else:
                    stddev = stddevVec

                stddev = tf.cast(stddev, dtype=tf.float32)/255.
                
                #label = tf.constant(LABEL_true_img) if stddev <1e-7 else tf.constant(LABEL_noisy_img)

                #adding_noise = tf.broadcast_to(stddev, tf.shape(x)[:2])[..., None]
                x_distorted += tf.random.normal(tf.shape(x_distorted), stddev=stddev, name="wgn", dtype=tf.float32)

            return x_distorted, gt, LABEL_noisy_img


        if status == 'validation':
            return DS.map(lambda x,y:tf.py_function(
                            func=train_adding_distortion_fn,
                            inp=[x, y, 
                                gt_noisy,
                                adding_noise, 
                                adding_blur,
                                adding_compression,
                                self.train_std_interval, 
                                self.train_blur_interval,
                                self.train_compres_interval], 
                            Tout=(tf.float32, tf.float32, tf.float32)
                        ),
                        num_parallel_calls=tf.data.AUTOTUNE,
                        deterministic=False
                        ) \
                    .prefetch(tf.data.AUTOTUNE)
        elif status == 'test':
            return {f'y_gibbs_{keep_fraction_i}_wgn_{stddev_i}': DS.map(
                                    lambda x,y: tf.py_function(
                                        func=train_adding_distortion_fn,
                                        inp=[x, y, 
                                            gt_noisy,
                                            adding_noise, 
                                            adding_blur,
                                            adding_compression,
                                            stddev_i, 
                                            keep_fraction_i,
                                            compressVec_i], 
                                        Tout=(tf.float32, tf.float32, tf.float32)
                                    ),
                                    num_parallel_calls=tf.data.AUTOTUNE,
                                    deterministic=False) \
                                    .prefetch(tf.data.AUTOTUNE).batch(1)
                                        for stddev_i in self.test_stdVec 
                                            for keep_fraction_i in self.test_blurVec
                                                for compressVec_i in self.test_compresVec 
            }

        elif status == 'test_random':
            return DS.map( lambda x,y: tf.py_function(
                                func=train_adding_distortion_fn,
                                    inp=[x, y, 
                                        gt_noisy,
                                        adding_noise, 
                                        adding_blur,
                                        adding_compression,
                                        self.validation_std_interval,
                                        self.train_blur_interval,
                                        self.train_compres_interval], 
                                    Tout=(tf.float32, tf.float32, tf.float32)
                                ),
                                num_parallel_calls=tf.data.AUTOTUNE,
                                deterministic=False) \
                                .prefetch(tf.data.AUTOTUNE)
                                        
        elif status == 'training':
            return DS.map(lambda x,y:tf.py_function(
                            func=train_adding_distortion_fn,
                            inp=[x, y, 
                                gt_noisy,
                                adding_noise, 
                                adding_blur,
                                adding_compression,
                                self.train_std_interval, 
                                self.train_blur_interval,
                                self.train_compres_interval], 
                            Tout=(tf.float32, tf.float32, tf.float32)
                        ),
                        num_parallel_calls=tf.data.AUTOTUNE,
                        deterministic=False) \
                    .prefetch(tf.data.AUTOTUNE).repeat(self.conf_train['repeats'])








def _resize_or_crop_fn(img, WH, channels_num, resize_crop_ratio = 0.5):
    def _apply_crop(img, w, h):
        if len(img.get_shape())>2:
            return tf.image.random_crop(value=img, 
                            size=(h, w, img.get_shape()[2]))
        else:
            return tf.image.random_crop(value=img, size=(h,w))

    def _apply_resize(img, w, h):
        return tf.image.resize(images=img, size=[h, w])

    w, h = WH

    if h <= int(resize_crop_ratio*img.get_shape()[0]):
        img = _apply_crop(img, img.get_shape()[1], h)
    elif h < img.get_shape()[0] and  h > int(resize_crop_ratio*img.get_shape()[0]):
        img = _apply_resize(img, img.get_shape()[1], h)
    elif h > img.get_shape()[0]:
        img = tf.pad(img, [[0, h-img.get_shape()[0]], [0, 0], [0, 0]])

    if w <= int(resize_crop_ratio*img.get_shape()[1]):
        img = _apply_crop(img, w, img.get_shape()[0])
    elif w < img.get_shape()[1] and  w > int(resize_crop_ratio*img.get_shape()[1]):
        img = _apply_resize(img, w, img.get_shape()[0])
    elif w > img.get_shape()[1]:
        img = tf.pad(img, [[0, 0], [0, w-img.get_shape()[1]], [0, 0]])

    if channels_num > img.get_shape()[2]:
        img = tf.repeat(img, repeats=channels_num, axis=2)

    return img[..., 0:channels_num]


def _data_augmentation(img, opt=None):
    if opt in tf.range(1, 5):
        return tf.image.rot90(img, k=opt)
    elif opt in tf.range(5, 9):
        return tf.experimental.numpy.flipud(tf.image.rot90(img, k=opt-3))





def _get_paired_image_fn(file_path: tf.Tensor, 
                    WH: int, 
                    channels_num: int,
                    gt_noisy:bool,
                    mode:str,
                    debug:bool
)->Union[tf.Tensor, tf.Tensor]:

    if gt_noisy and mode=='Training_valid':
        img_gt = _get_image(file_path, WH, channels_num, mode, debug)
        file_path = tf.strings.regex_replace(file_path, "GT", "NOISY")
        img_noisy = _get_image(file_path, WH, channels_num, mode, debug)
    else:
        img_noisy = _get_image(file_path, WH, channels_num, mode, debug)
        img_gt = img_noisy

    return img_noisy, img_gt





# get image
def _get_image(file_path: tf.Tensor, 
                    WH: int, 
                    channels_num: int,
                    mode:str,
                    debug:bool
)->tf.Tensor:

    file = tf.io.read_file(file_path)

    try:
        # if the class is jpeg, use `decode_jpeg` otherwise use decode_png
        if tf.strings.split(tf.strings.lower(file_path), sep='.')[-1] in ['jpeg', 'jpg']:
            img = tf.io.decode_jpeg(file)
        elif tf.strings.split(tf.strings.lower(file_path), sep='.')[-1] in ['bmp']:
            img = tf.image.decode_bmp(file)
        else:
            img = tf.image.decode_png(file, dtype=tf.uint8) 
    except:
        tf.print('bad file: ', file_path)
        if debug:
            return tf.zeros((WH[0], WH[1],channels_num), dtype=tf.float32)


    #tf.print(img.get_shape())
    if img.get_shape()[-1]>3:
        img = img[:,:,0:3]

    # sometimes there is an error in channel conversion with decode_png or decode_jpeg
    if channels_num == 1 and img.get_shape()[-1]>1:
        img = tf.image.rgb_to_grayscale(img)

    #img = tf.cond(
    #    tf.image.is_jpeg(file),
    #    lambda: tf.image.decode_jpeg(file, channels=channels_num),
    #    lambda: tf.image.decode_png(file, channels=channels_num))

    #convert unit8 tensor to floats in the [0,1]range
    img = tf.image.convert_image_dtype(img, tf.float32)

    if WH[0] > 0 and WH[1] > 0:
        img = _resize_or_crop_fn(img, WH, channels_num) 

    # the image dimension must be divisible to 16 for 4 scales
    block_sz = 16
    H_, W_= img.get_shape()[0] //block_sz, img.get_shape()[1] //block_sz
    if H_ > 0 and W_ > 0:
        img = _resize_or_crop_fn(img, [block_sz*W_, block_sz*H_], channels_num) 

    # augmentation is applied on squrae-shaped images only!
    opt = tf.random.uniform(shape=(), minval=1, maxval=9, dtype=tf.int32)
    if img.shape[0] == img.shape[1] and mode=='Training_valid':
        img = _data_augmentation(img, opt)

    return img






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
