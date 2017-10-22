import os.path
import tensorflow as tf

import helper

import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # DONE: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    input_w = graph.get_tensor_by_name(vgg_input_tensor_name)
    probabilities = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_w, probabilities, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # DONE: Implement function

    kernel_initializer = tf.truncated_normal_initializer(stddev=0.01)
    # FCN16
    # divide output stride in half by predicting from 16 pixel stride layer
    # add  1x1 convo layer on top of pool4 this adds additonal class predictions
    # fuse this output with conv7 at stride 32 by adding 2x upsampling layer and summing both predictions
    # initialize the 2x upsampling to bi-linear interpolations finally upsample stride 16 predictions to image size

    # FCN32
    # pool4 layer params are zero-initialized

    # FCN16
    # fusing predictions from pool3 with a 2x upsampling of predictions fused from pool4 and conv7

    # input, class or features so road or not road, kernel size 1 since it is only a 1x1 convolution, padding, and regularization

    layer_7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,
                                   padding='same',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # layer_7_1x1 = tf.layers.batch_normalization(layer_7_1x1)
    # layer_7_1x1 = tf.nn.relu(layer_7_1x1)

    # layer_4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
    #                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    #layer_3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
    #                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # upscale/ add features
    output = tf.layers.conv2d_transpose(layer_7_1x1, 512, 4, strides=(2, 2), padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # add layers together or "skip connections"
    output = tf.add(output, vgg_layer4_out)
    # upscale / reduce features
    output = tf.layers.conv2d_transpose(output, 256, 4, strides=(2, 2), padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # add layers together or "skip connections"
    output = tf.add(output, vgg_layer3_out)
    # upscale / reduce features
    output = tf.layers.conv2d_transpose(output, num_classes, 16, strides=(8, 8), padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                        kernel_initializer=kernel_initializer)

    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # DONE: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="my_logits")
    # reg_ws = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    # cross_entropy_loss += tf.reduce_sum(reg_ws)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(cross_entropy_loss)

    return logits, training_operation, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # DONE: Implement function
    sess.run(tf.global_variables_initializer())

    loss_log = []

    for epoch in range(epochs):
        print("Epoch {}...".format(epoch))
        epoch_loss = 0
        batch_count = 0

        for (image, label) in get_batches_fn(batch_size):
            feed = {input_image: image,
                    correct_label: label,
                    keep_prob: 0.5,
                    learning_rate: 0.0003}
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed)

            epoch_loss += loss
            batch_count += 1
            print("Loss: {}".format(loss))

        # add average loss in epoch
        loss_log.append(epoch_loss / batch_count)

    return loss_log

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    model_runs_dir = './runs/normal'
    model_pb_runs_dir = './runs/normal_pb'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    epochs = 18
    batch_size = 8

    # Create a TensorFlow configuration object. This will be
    # passed as an argument to the session.
    config = tf.ConfigProto()

    # JIT level, this can be set to ON_1 or ON_2
    jit_level = tf.OptimizerOptions.ON_1
    config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session(config=config) as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        correct_label = tf.placeholder(tf.int32)
        learning_rate = tf.placeholder(tf.float32)

        # DONE: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # DONE: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        loss_log = train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        # for i in tf.get_default_graph().get_operations():
        #     print(i.name)

        # exit()

        # DONE: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # Plot loss
        helper.plot_loss(runs_dir, loss_log)

        # OPTIONAL: Apply the trained model to a video
        saver = tf.train.Saver(tf.trainable_variables())
        save_path = os.path.join(model_runs_dir, 'model.ckpt')
        saver.save(sess, save_path)
        print('Saved normal at : {}'.format(save_path))

        save_path_pb = os.path.join(model_pb_runs_dir, 'graph.pb')
        save_path_pb_model = os.path.join(model_pb_runs_dir, 'test_model')

        # Save GraphDef
        tf.train.write_graph(sess.graph_def, '.', save_path_pb)
        # Save checkpoint
        saver.save(sess=sess, save_path=save_path_pb_model)
        print('Saved normal pb at : {}'.format(save_path_pb))

if __name__ == '__main__':
    run()