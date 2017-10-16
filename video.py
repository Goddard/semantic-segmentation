import os.path
import tensorflow as tf
import helper
import cv2
import freeze

from distutils.version import LooseVersion

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_ss(saver, sess, save_file):
    saver.restore(sess, save_file)
    # test_accuracy = sess.run(
    #     accuracy,
    #     feed_dict={features: mnist.test.images, labels: mnist.test.labels})

    # print('Test Accuracy: {}'.format(test_accuracy))

def out_layer(sess):
    vgg_tag = 'vgg16'
    # vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    # vgg_layer3_out_tensor_name = 'layer3_out:0'
    # vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer_output_tensor_name = 'my_output/conv2d_transpose:0'


    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    # input_w = graph.get_tensor_by_name(vgg_input_tensor_name)
    probabilities = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    # layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    # layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    output_layer = graph.get_tensor_by_name(vgg_layer_output_tensor_name)

    return probabilities, output_layer

def load_graph(graph_file, use_xla=False):
    jit_level = 0
    config = tf.ConfigProto()
    if use_xla:
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        ops = sess.graph.get_operations()
        n_ops = len(ops)
        return sess, sess.graph, ops

def run_frozen():
    runs_dir = './runs'
    data_dir = './data'
    save_path = os.path.join(runs_dir, '')
    video_path = os.path.join(data_dir, 'driving.mp4')

    # 640 368
    video_size = (1280, 736)  # 736 720
    # keep_prob = 0.3

    graph_file = './runs/frozen_model.pb'
    sess, graph, ops = load_graph(graph_file)
    # for i in ops:
    #     print(i.name)

    image_input = graph.get_tensor_by_name('image_input:0')
    probabilities = graph.get_tensor_by_name('keep_prob:0')
    output_layer = graph.get_tensor_by_name('my_output/conv2d_transpose:0')

    # second_output_layer = graph.get_tensor_by_name('my_second_output:0')
    # my_logits = graph.get_tensor_by_name('my_logits:0')

    logits = tf.reshape(output_layer, (-1, 2))

    cap = cv2.VideoCapture(video_path)
    while not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
        cv2.waitKey(1000)
        print("Wait for the header")

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            image = helper.test_video(sess, logits, probabilities, image_input, frame, video_size)
            cv2.imshow('video', image)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print(str(pos_frame) + " frames")
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break

def run_normal():
    with tf.Session() as sess:
        runs_dir = './runs'
        data_dir = './data'
        save_path = os.path.join(runs_dir, '')
        video_path = os.path.join(data_dir, 'driving.mp4')

        # 640 368
        video_size = (1280, 736) #736 720

        model_saver = tf.train.import_meta_graph(save_path + 'model.ckpt.meta')
        model_saver.restore(sess, tf.train.latest_checkpoint(save_path + ''))

        graph = tf.get_default_graph()
        image_input = graph.get_tensor_by_name('image_input:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        output_layer = graph.get_tensor_by_name('my_output/conv2d_transpose:0')
        second_output_layer = graph.get_tensor_by_name('my_second_output:0')
        my_logits = graph.get_tensor_by_name('my_logits:0')

        logits = tf.reshape(output_layer, (-1, 2))

        for i in tf.get_default_graph().get_operations():
            print(i.name)

        cap = cv2.VideoCapture(video_path)
        while not cap.isOpened():
            cap = cv2.VideoCapture(video_path)
            cv2.waitKey(1000)
            print("Wait for the header")

        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        while True:
            flag, frame = cap.read()
            if flag:
                # The frame is ready and already captured
                image = helper.test_video(sess, logits, keep_prob, image_input, frame, video_size)
                cv2.imshow('video', image)
                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                print(str(pos_frame) + " frames")
            else:
                # The next frame is not ready, so we try to read it again
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                print("frame is not ready")
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(1000)

            if cv2.waitKey(10) == 27:
                break

            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break

def run():
    # freeze.freeze_graph('./runs', "image_input,keep_prob,my_output/conv2d_transpose")

    run_normal()

if __name__ == '__main__':
    run()