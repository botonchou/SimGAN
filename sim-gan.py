#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implementation of `3.1 Appearance-based Gaze Estimation` from
[Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/pdf/1612.07828v1.pdf).
"""

import os
import sys
import cv2
import pprint

from keras import applications
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

from dlutils import plot_image_batch_w_labels
from utils.image_history_buffer import ImageHistoryBuffer

tf.flags.DEFINE_string("exp-dir", None, "Directory to write models to")
tf.flags.DEFINE_string("refiner-model-path", None, "Path to the refiner model .h5 file")
tf.flags.DEFINE_string("discriminator-model-path", None, "Path to the discriminator model .h5 file")
tf.flags.DEFINE_string("synthetic-data-dir", None, "Path to the synthesized data")
tf.flags.DEFINE_string("real-data-dir", None, "Path to the real data")

tf.flags.DEFINE_boolean("display", True, "display image")
tf.flags.DEFINE_boolean("shuffle", True, "shuffle images")

tf.flags.DEFINE_integer("log-every-nth", 100, "Log every nth iteration")
tf.flags.DEFINE_integer("max-iter", 100000, "maximum training iterations")
tf.flags.DEFINE_integer("start-iter", 0, "starting iteration count (temporary fix)")

tf.flags.DEFINE_integer("batch-size", 512, "size of mini-batch")
tf.flags.DEFINE_integer("kd", 1, "number of discriminator updates per step")
tf.flags.DEFINE_integer("kg", 2, "number of generative network updates per step")
tf.flags.DEFINE_float("learning-rate", 0.001, "Learning rate")
FLAGS = tf.flags.FLAGS

assert FLAGS.exp_dir is not None

FLAGS.model_dir = FLAGS.exp_dir + "/models/"
FLAGS.vis_dir = FLAGS.exp_dir + "/vis/"

# image dimensions
img_width = 55
img_height = 35
img_channels = 1

# training params
batch_size = FLAGS.batch_size
log_interval = FLAGS.log_every_nth

def get_image_batch(generator):
    """keras generators may generate an incomplete batch for the last batch"""
    img_batch = generator.next()
    if len(img_batch) != batch_size:
        img_batch = generator.next()

    assert len(img_batch) == batch_size

    return img_batch

def refiner_network(input_image_tensor):
    """
    The refiner network, Rθ, is a residual network (ResNet). It modifies the synthetic image on a pixel level, rather
    than holistically modifying the image content, preserving the global structure and annotations.

    :param input_image_tensor: Input tensor that corresponds to a synthetic image.
    :return: Output tensor that corresponds to a refined synthetic image.
    """
    def resnet_block(input_features, nb_features=64, nb_kernel_rows=3, nb_kernel_cols=3):
        """
        A ResNet block with two `nb_kernel_rows` x `nb_kernel_cols` convolutional layers,
        each with `nb_features` feature maps.

        See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.

        :param input_features: Input tensor to ResNet block.
        :return: Output tensor from ResNet block.
        """
        y = layers.Convolution2D(nb_features, nb_kernel_rows, nb_kernel_cols, border_mode='same')(input_features)
        y = layers.Activation('relu')(y)
        y = layers.Convolution2D(nb_features, nb_kernel_rows, nb_kernel_cols, border_mode='same')(y)

        y = layers.merge([input_features, y], mode='sum')
        return layers.Activation('relu')(y)

    # an input image of size w × h is convolved with 3 × 3 filters that output 64 feature maps
    x = layers.Convolution2D(64, 3, 3, border_mode='same', activation='relu')(input_image_tensor)

    # the output is passed through 4 ResNet blocks
    for _ in range(4):
        x = resnet_block(x)

    # the output of the last ResNet block is passed to a 1 × 1 convolutional layer producing 1 feature map
    # corresponding to the refined synthetic image
    return layers.Convolution2D(1, 1, 1, border_mode='same', activation='tanh')(x)


def discriminator_network(input_image_tensor):
    """
    The discriminator network, Dφ, contains 5 convolution layers and 2 max-pooling layers.

    :param input_image_tensor: Input tensor corresponding to an image, either real or refined.
    :return: Output tensor that corresponds to the probability of whether an image is real or refined.
    """
    x = layers.Convolution2D(96, 3, 3, border_mode='same', subsample=(2, 2), activation='relu')(input_image_tensor)
    x = layers.Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), border_mode='same', strides=(1, 1))(x)
    x = layers.Convolution2D(32, 3, 3, border_mode='same', subsample=(1, 1), activation='relu')(x)
    x = layers.Convolution2D(32, 1, 1, border_mode='same', subsample=(1, 1), activation='relu')(x)
    x = layers.Convolution2D(2, 1, 1, border_mode='same', subsample=(1, 1), activation='relu')(x)

    # here one feature map corresponds to `is_real` and the other to `is_refined`,
    # and the custom loss function is then `tf.nn.sparse_softmax_cross_entropy_with_logits`
    return layers.Reshape((-1, 2))(x)

def get_data_generator(synthetic_data_dir, real_data_dir):

    datagen = image.ImageDataGenerator(
        preprocessing_function=applications.xception.preprocess_input,
        dim_ordering='tf'
    )

    flow_from_directory_params = {
        'target_size': (img_height, img_width),
        'color_mode': 'grayscale' if img_channels == 1 else 'rgb',
        'class_mode': None,
        'batch_size': batch_size
    }

    synthetic_generator = datagen.flow_from_directory(
        directory=synthetic_data_dir,
        shuffle=FLAGS.shuffle,
        **flow_from_directory_params
    )

    real_generator = datagen.flow_from_directory(
        directory=real_data_dir,
        shuffle=FLAGS.shuffle,
        **flow_from_directory_params
    )

    return synthetic_generator, real_generator

def pretrain_refiner(refiner_model, synthetic_generator):
    # we first train the Rθ network with just self-regularization loss for 1,000 steps
    print 'pre-training the refiner network...'
    gen_loss = np.zeros(shape=len(refiner_model.metrics_names))

    for i in range(1000):
        synthetic_image_batch = get_image_batch(synthetic_generator)
        gen_loss = np.add(refiner_model.train_on_batch(synthetic_image_batch, synthetic_image_batch), gen_loss)

        # log every `log_interval` steps
        if i % log_interval == 0:

            if FLAGS.display:
                fig_name = 'refined_image_batch_pre_train_step_{}.png'.format(i)
                print 'Saving batch of refined images during pre-training at step: {}.'.format(i)
                visualize_training_procedure(fig_name, synthetic_generator, refiner_model)

            print 'Refiner model self regularization loss: {}.'.format(gen_loss / log_interval)
            gen_loss = np.zeros(shape=len(refiner_model.metrics_names))

def pretrain_discriminator(discriminator_model, real_generator, synthetic_generator):
    # and Dφ for 200 steps (one mini-batch for refined images, another for real)
    print 'pre-training the discriminator network...'
    disc_loss = np.zeros(shape=len(discriminator_model.metrics_names))

    for _ in range(100):
        real_image_batch = get_image_batch(real_generator)
        disc_loss = np.add(discriminator_model.train_on_batch(real_image_batch, y_real), disc_loss)

        synthetic_image_batch = get_image_batch(synthetic_generator)
        refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)
        disc_loss = np.add(discriminator_model.train_on_batch(refined_image_batch, y_refined), disc_loss)

    # hard-coded for now
    print 'Discriminator model loss: {}.'.format(disc_loss / (100 * 2))

def adversarial_training(synthetic_data_dir, real_data_dir, refiner_model_path=None, discriminator_model_path=None):
    """Adversarial training of refiner network Rθ and discriminator network Dφ."""

    # Define model input and output tensors
    synthetic_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    refined_image_tensor = refiner_network(synthetic_image_tensor)

    refined_or_real_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    discriminator_output = discriminator_network(refined_or_real_image_tensor)

    # Define models
    refiner_model = models.Model(
        input=synthetic_image_tensor, output=refined_image_tensor,
        name='refiner'
    )
    discriminator_model = models.Model(
        input=refined_or_real_image_tensor, output=discriminator_output,
        name='discriminator'
    )

    # Combined must output the refined image along w/ the disc's classification of it for the refiner's self-reg loss
    refiner_model_output = refiner_model(synthetic_image_tensor)
    combined_output = discriminator_model(refiner_model_output)
    combined_model = models.Model(
        input=synthetic_image_tensor,
        output=[refiner_model_output, combined_output],
        name='combined'
    )

    discriminator_model_output_shape = discriminator_model.output_shape

    """
    print refiner_model.summary()
    print discriminator_model.summary()
    print combined_model.summary()
    """

    # Define custom l1 loss function for the refiner
    def self_regularization_loss(y_true, y_pred):
        delta = 0.0001  # FIXME: need to figure out an appropriate value for this
        return tf.multiply(delta, tf.reduce_sum(tf.abs(y_pred - y_true)))

    # Define custom local adversarial loss (softmax for each image section) for the discriminator
    # the adversarial loss function is the sum of the cross-entropy losses over the local patches
    def local_adversarial_loss(y_true, y_pred):
        # y_true and y_pred have shape (batch_size, # of local patches, 2), but really we just want to average over
        # the local patches and batch size so we can reshape to (batch_size * # of local patches, 2)
        y_true = tf.reshape(y_true, (-1, 2))
        y_pred = tf.reshape(y_pred, (-1, 2))
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        return tf.reduce_mean(loss)

    # Compile models
    sgd = optimizers.SGD(lr=FLAGS.learning_rate)

    refiner_model.compile(optimizer=sgd, loss=self_regularization_loss)
    discriminator_model.compile(optimizer=sgd, loss=local_adversarial_loss)
    discriminator_model.trainable = False
    combined_model.compile(optimizer=sgd, loss=[self_regularization_loss, local_adversarial_loss])

    synthetic_generator, real_generator = get_data_generator(synthetic_data_dir, real_data_dir)

    # TODO Should use one-sided smoothing suggested by Goodfellow
    # the target labels for the cross-entropy loss layer are 0 for every yj (real) and 1 for every xi (refined)
    y_real = np.array([[[1.0, 0.0]] * discriminator_model_output_shape[1]] * batch_size)
    y_refined = np.array([[[0.0, 1.0]] * discriminator_model_output_shape[1]] * batch_size)
    assert y_real.shape == (batch_size, discriminator_model_output_shape[1], 2)

    if refiner_model_path is None:
        pretrain_refiner(refiner_model, synthetic_generator)
        refiner_model.save(FLAGS.model_dir + 'refiner_model_pre_trained.h5')
    else:
        tf.logging.info("Loading refiner model from {}".format(FLAGS.model_dir + refiner_model_path))
        refiner_model.load_weights(FLAGS.model_dir + refiner_model_path)

    if discriminator_model_path is None:
        pretrain_discriminator(discriminator_model, real_generator, synthetic_generator)
        discriminator_model.save(FLAGS.model_dir + 'discriminator_model_pre_trained.h5')
    else:
        tf.logging.info("Loading discriminator model from {}".format(FLAGS.model_dir + discriminator_model_path))
        discriminator_model.load_weights(FLAGS.model_dir + discriminator_model_path)

    # TODO: what is an appropriate size for the image history buffer?
    image_history_buffer = ImageHistoryBuffer((0, img_height, img_width, img_channels), batch_size * 100, batch_size)

    combined_loss = np.zeros(shape=len(combined_model.metrics_names))
    disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
    disc_loss_refined = np.zeros(shape=len(discriminator_model.metrics_names))

    # see Algorithm 1 in https://arxiv.org/pdf/1612.07828v1.pdf
    for i in range(FLAGS.start_iter, FLAGS.start_iter + FLAGS.max_iter):
        print 'Step: {} of {}.'.format(i, FLAGS.start_iter + FLAGS.max_iter)

        # train the refiner
        for _ in range(FLAGS.kg * 2):
            # sample a mini-batch of synthetic images
            synthetic_image_batch = get_image_batch(synthetic_generator)

            # update θ by taking an SGD step on mini-batch loss LR(θ)
            combined_loss = np.add(combined_model.train_on_batch(synthetic_image_batch,
                                                                 [synthetic_image_batch, y_real]), combined_loss)

        for _ in range(FLAGS.kd):
            # sample a mini-batch of synthetic and real images
            synthetic_image_batch = get_image_batch(synthetic_generator)
            real_image_batch = get_image_batch(real_generator)

            # refine the synthetic images w/ the current refiner
            refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)

            # use a history of refined images
            half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
            image_history_buffer.add_to_image_history_buffer(refined_image_batch)

            if len(half_batch_from_image_history):
                refined_image_batch[:batch_size // 2] = half_batch_from_image_history

            # update φ by taking an SGD step on mini-batch loss LD(φ)
            disc_loss_real = np.add(discriminator_model.train_on_batch(real_image_batch, y_real), disc_loss_real)
            disc_loss_refined = np.add(discriminator_model.train_on_batch(refined_image_batch, y_refined),
                                       disc_loss_refined)

        if i % log_interval == 0:
            # plot batch of refined images w/ current refiner
            if FLAGS.display:
                fig_name = 'refined_image_batch_step_{}.png'.format(i)
                print 'Saving batch of refined images at adversarial step: {}.'.format(i)
                visualize_training_procedure(fig_name, synthetic_generator, refiner_model)

            # log loss summary
            print 'Refiner model loss: {}.'.format(combined_loss / (log_interval * FLAGS.kg * 2))
            print 'Discriminator model loss real: {}.'.format(disc_loss_real / (log_interval * FLAGS.kd * 2))
            print 'Discriminator model loss refined: {}.'.format(disc_loss_refined / (log_interval * FLAGS.kd * 2))

            combined_loss = np.zeros(shape=len(combined_model.metrics_names))
            disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
            disc_loss_refined = np.zeros(shape=len(discriminator_model.metrics_names))

            # save model checkpoints
            """
            model_checkpoint_base_name = FLAGS.model_dir + '{}_model_step_{}.h5'
            refiner_model.save(model_checkpoint_base_name.format('refiner', i))
            discriminator_model.save(model_checkpoint_base_name.format('discriminator', i))
            """

def put_images_on_grid(images):

    n_images_per_row = 32
    nb_columns = n_images_per_row
    nb_rows = int(np.ceil(float(batch_size) / n_images_per_row))

    # Turn (N, height, width) to (N, height, width, 1) for consistency
    if images.ndim == 3:
        images = images[..., None]

    N, H, W, C = images.shape

    grid = np.zeros((nb_rows * H, nb_columns * W, C), dtype=images.dtype)

    for idx, image in enumerate(images):
        i = idx // n_images_per_row
        j = idx %  n_images_per_row
        grid[i*H:(i+1)*H, j*W:(j+1)*W, ...] = image

    # Normalize image based on min/max of images
    minimum, maximum = np.min(images), np.max(images)
    tf.logging.info("np.min(images) = {}, np.max(images) = {}".format(minimum, maximum))
    grid = (grid - minimum) / (maximum - minimum)
    grid = np.clip(grid * 255., 0, 255).astype(np.uint8)

    return grid

def visualize_training_procedure(fig_name, synthetic_generator, refiner_model):
    fig_name = FLAGS.vis_dir + fig_name

    synthetic_images = get_image_batch(synthetic_generator)
    refined_images = refiner_model.predict_on_batch(synthetic_images)

    simages = put_images_on_grid(synthetic_images)
    divider = np.zeros((10, simages.shape[1], simages.shape[2]), dtype=np.uint8)
    rimages = put_images_on_grid(refined_images)

    fig = np.concatenate([simages, divider, rimages])

    cv2.imwrite(fig_name, fig)

def main(synthetic_data_dir, real_data_dir,
         refiner_model_path, discriminator_model_path):

    exp_config = pprint.pformat(FLAGS.__flags)
    tf.logging.info(exp_config)

    adversarial_training(synthetic_data_dir, real_data_dir,
                         refiner_model_path, discriminator_model_path)

if __name__ == '__main__':
    # TODO: if pre-trained models are passed in, we don't take the steps they've been trained for into account
    main(
        FLAGS.synthetic_data_dir, FLAGS.real_data_dir,
        FLAGS.refiner_model_path, FLAGS.discriminator_model_path
    )
