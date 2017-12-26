# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from preprocessing import preprocessing_factory
import reader
import model
import time
import os
from os import listdir
from os.path import isfile,join
import sys

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string("model_file", "models.ckpt", "")
tf.app.flags.DEFINE_string("image_file", "test", "")

FLAGS = tf.app.flags.FLAGS


def gen(imagename, png = False):

    # Get image's height and width.
    height = 0
    width = 0
    with open(imagename, 'rb') as img:
        with tf.Session().as_default() as sess:
            if png:
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:

            # Read image data.
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            image = reader.get_image(imagename, height, width, image_preprocessing_fn)

            # Add batch dimension
            image = tf.expand_dims(image, 0)

            generated = model.net(image, training=False)
            generated = tf.cast(generated, tf.uint8)

            # Remove batch dimension
            generated = tf.squeeze(generated, [0])

            # Restore model variables.
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Use absolute path
            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)

            # Make sure 'generated' directory exists.
            generated_file = 'generated/' + imagename.split("/")[-2]+'/'+imagename[imagename.rfind("/")+1:]
            print(imagename)
            print(generated_file)
            if os.path.exists('generated') is False:
                os.makedirs('generated')
            if os.path.exists( 'generated/' + imagename.split("/")[-2]) is False:
                os.makedirs('generated/' + imagename.split("/")[-2])

            # Generate and write image data to file.
            with open(generated_file, 'wb') as img:
                start_time = time.time()
                img.write(sess.run(tf.image.encode_png(generated)))
                end_time = time.time()
                tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

                tf.logging.info('Done. Please check %s.' % generated_file)

def main(_):
    filenames = [join(FLAGS.image_file, f) for f in listdir(FLAGS.image_file) if isfile(join(FLAGS.image_file, f))]
    # print(filenames)
    for imagename in filenames:
        if imagename.lower().endswith("png"):
 
            gen(imagename, png = True)
        elif imagename.lower().endswith("jpg"):


            gen(imagename, png = False)
            

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
