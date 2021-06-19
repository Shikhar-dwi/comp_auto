"""
requires TFC v2 (`pip install tensorflow-compression==2.*`).
"""

import argparse
import glob
import sys
import numpy as np
import os
from absl import app
from absl.flags import argparse_flags
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_compression as tfc



class Encode7(tf.keras.Sequential):
  """The Encoder"""
  def __init__(self):
    super().__init__(name="Encoder7")
    self.add(tf.keras.layers.Lambda(lambda x: x / 255.))
    self.add(tfc.SignalConv2D(
        64, (7, 7), name="layer7_0", corr=True, strides_down=4,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn7_0", inverse=False)))

class Encode5(tf.keras.Sequential):
  """The Encoder"""
  def __init__(self):
    super().__init__(name="Encoder5")
    self.add(tf.keras.layers.Lambda(lambda x: x / 255.))
    self.add(tfc.SignalConv2D(
        64, (5, 5), name="layer5_0", corr=True, strides_down=4,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn5_0", inverse=False)))
        
class Encode3(tf.keras.Sequential):
  """The Encoder"""
  def __init__(self):
    super().__init__(name="Encoder3")
    self.add(tf.keras.layers.Lambda(lambda x: x / 255.))
    self.add(tfc.SignalConv2D(
        64, (3, 3), name="layer3_0", corr=True, strides_down=4,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn3_0", inverse=False)))

class Encoder(tf.keras.Sequential):
  """The Encoder"""

  def __init__(self, num_filters):
    super().__init__(name="Encoder")
    self.add(tfc.SignalConv2D(
        128, (5, 5), name="layer_0", corr=True, strides_down=4,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_0")))
    self.add(tfc.SignalConv2D(
        128, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_1")))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=False,
        activation=tfc.GDN(name="gdn_2")))


class Decoder(tf.keras.Sequential):
  """The Decoder."""

  def __init__(self, num_filters):
    super().__init__(name="Decoder")
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_0", inverse=True)))
    self.add(tfc.SignalConv2D(
        128, (5, 5), name="layer_1", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_1", inverse=True)))
    self.add(tfc.SignalConv2D(
        128, (5, 5), name="layer_3", corr=False, strides_up=4,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_2", inverse=True)))

class Decode7(tf.keras.Sequential):
  """The Decoder."""
  def __init__(self):
    super().__init__(name="Decoder7")
    self.add(tfc.SignalConv2D(
        32, (7, 7), name="layer7_1", corr=False, strides_up=4,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn7_0", inverse=True)))
        
class Decode5(tf.keras.Sequential):
  """The Decoder."""
  def __init__(self):
    super().__init__(name="Decoder7")
    self.add(tfc.SignalConv2D(
        32, (5, 5), name="layer5_1", corr=False, strides_up=4,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn5_0", inverse=True)))
        
class Decode3(tf.keras.Sequential):
  """The Decoder."""
  def __init__(self):
    super().__init__(name="Decoder7")
    self.add(tfc.SignalConv2D(
        32, (3, 3), name="layer3_1", corr=False, strides_up=4,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn3_0", inverse=True)))
        
class Decoder_final(tf.keras.Sequential):
  """The Decoder."""
  def __init__(self):
    super().__init__(name="Decoder_final")
    self.add(tfc.SignalConv2D(
        3, (5, 5), name="layerf_6", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True,
        activation=None))
    self.add(tf.keras.layers.Lambda(lambda x: x * 255.))



class comp_auto(tf.keras.Model):
  """Main model class."""

  def __init__(self, lmbda, num_filters):
    super().__init__()
    self.lmbda = lmbda
    self.Encoder = Encoder(num_filters)
    self.Encode7 = Encode7()
    self.Encode5 = Encode5()
    self.Encode3 = Encode3()
    self.decoder = Decoder(num_filters)
    self.Decode7 = Decode7()
    self.Decode5 = Decode5()
    self.Decode3 = Decode3()
    self.Definal = Decoder_final()
    self.prior = tfc.NoisyDeepFactorized(batch_shape=(num_filters,))
    self.build((None, None, None, 3))

  def call(self, x, training):
    """Computes rate and distortion losses."""
    entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.prior, coding_rank=3, compression=False)
    
    y5 = self.Encode5(x)
    y7 = self.Encode7(x)
    y3 = self.Encode3(x)
    y1 = tf.keras.layers.concatenate( [y7, y5 ,y3] )
    y = self.Encoder(y1)
    
    y_hat, bits = entropy_model(y, training=training)
    
    x_hat = self.decoder(y_hat)
    z5 = self.Decode7(x_hat)
    z7 = self.Decode5(x_hat)
    z3 = self.Decode3(x_hat)
    z1 = tf.keras.layers.concatenate( [z7, z5 ,z3] )
    x_hat = self.Definal(z1)
    
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
    bpp = tf.reduce_sum(bits) / num_pixels
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    # ms_ssim =1- tf.squeeze((tf.image.ssim_multiscale(x, x_hat, 255)))
    # The rate-distortion trade-off.
    loss = bpp + self.lmbda * mse #+ ms_ssim
    return loss, bpp, mse

  def train_step(self, x):
    with tf.GradientTape() as tape:
      loss, bpp, mse = self(x, training=True)
    variables = self.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    # self.mse.update_state(ms_ssim)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

  def test_step(self, x):
    loss, bpp, mse = self(x, training=False)
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    # self.mse.update_state(ms_ssim)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

  def predict_step(self, x):
    raise NotImplementedError("Prediction API is not supported.")

  def compile(self, **kwargs):
    super().compile(
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    )
    self.loss = tf.keras.metrics.Mean(name="loss")
    self.bpp = tf.keras.metrics.Mean(name="bpp")
    self.mse = tf.keras.metrics.Mean(name="mse")
    # self.ms_ssim = tf.keras.metrics.Mean(name="ms_ssim")

  def fit(self, *args, **kwargs):
    retval = super().fit(*args, **kwargs)
    # After training, fix range coding tables.
    self.entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.prior, coding_rank=3, compression=True)
    return retval

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
  ])
  def compress(self, x):
    """Compresses an image."""
    # Add batch dimension and cast to float.
    x = tf.expand_dims(x, 0)
    x = tf.cast(x, dtype=tf.float32)
    
    y5 = self.Encode5(x)
    y7 = self.Encode7(x)
    y3 = self.Encode3(x)
    y1 = tf.keras.layers.concatenate( [y7, y5 ,y3] )
    y = self.Encoder(y1)
    
    # Including shapes of both image and bottleneck layer tensor.
    x_shape = tf.shape(x)[1:-1]
    y_shape = tf.shape(y)[1:-1]
    return self.entropy_model.compress(y), x_shape, y_shape

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(1,), dtype=tf.string),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
  ])
  
  def decompress(self, string, x_shape, y_shape):
    """Decompresses an image."""
    y_hat = self.entropy_model.decompress(string, y_shape)
    
    x_hat = self.decoder(y_hat)
    z5 = self.Decode7(x_hat)
    z7 = self.Decode5(x_hat)
    z3 = self.Decode3(x_hat)
    z1 = tf.keras.layers.concatenate( [z7, z5 ,z3] )
    x_hat = self.Definal(z1)
    
    x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
    # x_hat = x_hat * 255
    return tf.saturate_cast(tf.round(x_hat), tf.uint8)# Then casting to integer.

def read_png(filename):
  """Loads a PNG image file."""
  string = tf.io.read_file(filename)
  return tf.image.decode_image(string, channels=3)


def write_png(filename, image):
  """Saves an image to a PNG file."""
  string = tf.image.encode_png(image)
  tf.io.write_file(filename, string)
 
def check_image_size(image, patch_size):
  shape = tf.shape(image)
  return shape[0] >= patch_size and shape[1] >= patch_size and shape[-1] == 3

def crop_image(image, patch_size):
  image = tf.image.random_crop(image, (patch_size, patch_size, 3))
  return tf.cast(image, tf.float32)


def get_dataset(name, split, args):
  """Creates input data pipeline from a TF Datasets dataset."""
  with tf.device("/cpu:0"):
    dataset = tfds.load(name, split=split, shuffle_files=True)
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.filter(
        lambda x: check_image_size(x["image"], args.patch_size))
    dataset = dataset.map(
        lambda x: crop_image(x["image"], args.patch_size))
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
  return dataset



def train(args):
  """training the model."""

  model = comp_auto(args.lmbda, args.num_filters)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),#learning_rate=1e-4
  )
#   model = tf.keras.models.load_model(args.model_path+"/"+str(args.lmbda)+"_"+str(args.num_filters))
#   model.load_weights("weights/"+'a_model_weights'+str(args.lmbda)+"_"+str(args.num_filters)+'.h5')


  train_dataset = get_dataset("clic", "train", args)
  validation_dataset = get_dataset("clic", "validation", args)
  validation_dataset = validation_dataset.take(args.max_validation_steps)

  model.fit(
      train_dataset.prefetch(8),
      epochs=args.epochs,
      steps_per_epoch=args.steps_per_epoch,
      validation_data=validation_dataset.cache(),
      validation_freq=1,
    #   callbacks=[
    #       tf.keras.callbacks.TerminateOnNaN(),
    #       tf.keras.callbacks.TensorBoard(
    #           log_dir=args.train_path,
    #           histogram_freq=1, update_freq="epoch"),
    #       tf.keras.callbacks.experimental.BackupAndRestore(args.train_path),
    #   ],
      verbose=int(args.verbose),
  )
  model.save_weights("weights/"+'a_model_weights'+str(args.lmbda)+"_"+str(args.num_filters)+'.h5')
  model.save(args.model_path +"/"+str(args.lmbda)+"_"+str(args.num_filters))


def compress(args):
  """Compresses an image."""
  # Loading model and using it to compress the image.
  model = tf.keras.models.load_model(args.model_path+"/"+str(args.lmbda)+"_"+str(args.num_filters)) 
  #+"/"+str(args.lmbda)+"_"+str(args.num_filters) is added to ensure that different Lambda values get stored in different location.
  
  x = read_png(args.input_file)
  tensors = model.compress(x)

  # Packing the tensor with the shape information
  packed = tfc.PackedTensors()
  packed.pack(tensors)
  # Writing the tensor
  with open(args.output_file, "wb") as f:
    f.write(packed.string)

  x_hat = model.decompress(*tensors)
  
  x = tf.cast(x, tf.float32)
  x_hat = tf.cast(x_hat, tf.float32)
  mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
  psnr = tf.squeeze(tf.image.psnr(x, x_hat, 255))
  msssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255))
  msssim_db = -10. * tf.math.log(1 - msssim) / tf.math.log(10.)
  
  num_pixels = tf.reduce_prod(tf.shape(x)[:-1])
  bpp = len(packed.string) * 8 / num_pixels

  print(f"Mean squared error: {mse:0.4f}")
  print(f"PSNR (dB): {psnr:0.2f}")
  print(f"Multiscale SSIM: {msssim:0.4f}")
  print(f"Multiscale SSIM (dB): {msssim_db:0.2f}")
  print(f"Bits per pixel: {bpp:0.4f}")


def decompress(args):
  """Decompresses an image."""
  # Load the model and determine the dtypes of tensors required to decompress.
  model = tf.keras.models.load_model(args.model_path+"/"+str(args.lmbda)+"_"+str(args.num_filters))#
  dtypes = [t.dtype for t in model.decompress.input_signature]

  # unpacking the compressed file
  with open(args.input_file, "rb") as f:
    packed = tfc.PackedTensors(f.read())
  tensors = packed.unpack(dtypes)
  x_hat = model.decompress(*tensors)

  write_png(args.output_file, x_hat)

def testcomp(args):
  """Compresses an image set."""
  # Loading model and using it to calculate metrics for folder of images.
  model = tf.keras.models.load_model(args.model_path+"/"+str(args.lmbda)+"_"+str(args.num_filters))#+"/"+str(args.lmbda)+"_"+str(args.num_filters)
  n = os.listdir(args.test_glob) #List of training images
  n = sorted(n)
  mse  = np.zeros(len(n)).astype('float')
  psnr  = np.zeros(len(n)).astype('float')
  msssim  = np.zeros(len(n)).astype('float')
  bpp  = np.zeros(len(n)).astype('float')
  for i in range(0, len(n)): #initially from 0 to n 
            x = read_png(args.test_glob+'/'+n[i])
            
            tensors = model.compress(x)
            packed = tfc.PackedTensors()
            packed.pack(tensors)
            
            x_hat = model.decompress(*tensors)
            
            x = tf.cast(x, tf.float32)
            x_hat = tf.cast(x_hat, tf.float32)
            mse[i] = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
            psnr[i] = tf.squeeze(tf.image.psnr(x, x_hat, 255))
            msssim[i] = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255))
            num_pixels = tf.reduce_prod(tf.shape(x)[:-1])
            bpp[i] = len(packed.string) * 8 / num_pixels
            print(mse[i], psnr[i] ,msssim[i], bpp[i])
            
  print(sum(mse), sum(psnr) ,sum(msssim), sum(bpp))
  print(np.mean(mse),np.mean(psnr),np.mean(msssim),np.mean(bpp))



def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report progress and metrics when training, compressing or decompressing")
  parser.add_argument(
      "--model_path", default="modelcomp",
      help="Path where to save/load the trained model.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains the model")
  train_cmd.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--num_filters", type=int, default=192,
      help="Number of filters in bottleneck layer.")
  train_cmd.add_argument(
      "--lr", type=float, default=0.0001,
      help="Learning rate for training and validation.")
  train_cmd.add_argument(
      "--train_path", default="/tmp/saved_codegdn",
      help="Path where to log training metrics ")
  train_cmd.add_argument(
      "--batch_size", type=int, default=8,
      help="Batch size for training and validation.")
  train_cmd.add_argument(
      "--patch_size", type=int, default=128,
      help="Size of image patches for training and validation.")
  train_cmd.add_argument(
      "--epochs", type=int, default=60,
      help="Train up to this number of epochs.")
  train_cmd.add_argument(
      "--steps_per_epoch", type=int, default=25,
      help="Perform validation and produce logs after this many batches.")
  train_cmd.add_argument(
      "--max_validation_steps", type=int, default=16,
      help="Maximum number of batches to use for validation.")


  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes it")
  compress_cmd.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  compress_cmd.add_argument(
      "--num_filters", type=int, default=192,
      help="Number of filters in bottleneck layer.")
      
   # 'testcomp' subcommand.
  testcomp_cmd = subparsers.add_parser(
      "testcomp",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG files, compresses it, decompresses it and produce metrics.")
  testcomp_cmd.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  testcomp_cmd.add_argument(
      "--num_filters", type=int, default=192,
      help="Number of filters in bottleneck layer.")
  testcomp_cmd.add_argument(
      "--test_glob", type=str, default=None,
      help="Path for test folder on which metics need to be calculated ( group average )  ")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a compressed file, reconstructs the image, and writes back a PNG file.")
  decompress_cmd.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  decompress_cmd.add_argument(
      "--num_filters", type=int, default=192,
      help="Number of filters in bottleneck layer.")

  # Arguments for both 'compress' and 'decompress'.
  for cmd, ext in ((compress_cmd, ".comp"), (decompress_cmd, ".png")):
    cmd.add_argument(
        "input_file",
        help="Input filename.")
    cmd.add_argument(
        "output_file", nargs="?",
        help=f"Output filename (optional). If not provided, appends '{ext}' to "
             f"the input filename.")

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Invoke subcommand.
  if args.command == "train":
    train(args)
  elif args.command == "compress":
    if not args.output_file:
      args.output_file = args.input_file + ".comp"
    compress(args)
  elif args.command == "decompress":
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decompress(args)
  elif args.command == "testcomp":
       testcomp(args)


if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
