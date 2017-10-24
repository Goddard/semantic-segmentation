import os.path
import tensorflow as tf
import helper
import warnings

import cv2
import timeit

import graph_utils

from distutils.version import LooseVersion

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def run_video(sess, image_input, keep_prob, my_logits, data_dir):
    video_path = os.path.join(data_dir, 'driving.mp4')
    # 640 368
    up_s_vid_size = (1280, 720)
    down_s_vid_size = (576, 160)
    # video_size = (576, 160) #736 720 1280, 736

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
            start_time = timeit.default_timer()
            image = cv2.resize(frame, down_s_vid_size, interpolation=cv2.INTER_LINEAR)
            print("Downscale : {}".format(timeit.default_timer() - start_time))

            start_time = timeit.default_timer()
            image = helper.test_video(sess, my_logits, keep_prob, image_input, image, down_s_vid_size, up_s_vid_size)
            print("SS Test : {}".format(timeit.default_timer() - start_time))

            start_time = timeit.default_timer()
            image = cv2.resize(image, up_s_vid_size, interpolation=cv2.INTER_LINEAR)
            print("Upscale : {}".format(timeit.default_timer() - start_time))

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

def load_graph2(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

def load_model(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def run_altered(runs_dir):
    data_dir = './data'
    image_shape = (160, 576)
    model_location = os.path.join(runs_dir, 'model.pb')

    # sess, ops = graph_utils.load_graph(model_location, True)
    # graph = load_model(model_location)

    # with tf.Session(graph=sess.graph) as sess:
        # tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    config = tf.ConfigProto()
    jit_level = tf.OptimizerOptions.ON_1
    config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(model_location, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        # ops = sess.graph.get_operations()
        # n_ops = len(ops)

        image_input = sess.graph.get_tensor_by_name('image_input:0')
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        my_logits = sess.graph.get_tensor_by_name('my_logits:0')

        # sess.run(tf.global_variables_initializer())

        # tf.global_variables_initializer()

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, my_logits, keep_prob, image_input)

        run_video(sess, image_input, keep_prob, my_logits, data_dir)

def run_normal(runs_dir):
    with tf.Session() as sess:
        save_path = os.path.join(runs_dir, '')
        data_dir = './data'

        model_saver = tf.train.import_meta_graph(save_path + 'model.ckpt.meta')
        model_saver.restore(sess, tf.train.latest_checkpoint(save_path + ''))

        graph = tf.get_default_graph()
        image_input = graph.get_tensor_by_name('image_input:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        my_logits = graph.get_tensor_by_name('my_logits:0')

        model_pb_runs_dir = './runs/normal_pb'
        output_node_names = 'image_input,keep_prob,my_logits'
        save_path_pb = os.path.join(model_pb_runs_dir, 'model.pb')
        input_graph_def = graph.as_graph_def()
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")
        )

        with tf.gfile.GFile(save_path_pb, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        # Save GraphDef
        # tf.train.write_graph(sess.graph_def, '.', save_path_pb, as_text=False)
        print('Saved normal pb at : {}'.format(save_path_pb))

        run_video(sess, image_input, keep_prob, my_logits, data_dir)

def run():
    normal_runs_dir = os.path.join('./runs/normal', '')
    normal_pb_runs_dir = os.path.join('./runs/normal_pb', '')
    freeze_runs_dir = os.path.join('./runs/freeze', '')
    optimize_runs_dir = os.path.join('./runs/optimized', '')
    eight_bit_runs_dir = os.path.join('./runs/eight_bit', '')

    start_time = timeit.default_timer()
    # runs but still slow and doesn't detect well
    run_normal(normal_runs_dir)
    print("Normal : {}".format(timeit.default_timer() - start_time))

    # start_time = timeit.default_timer()
    # # run normal pb
    # run_altered(normal_pb_runs_dir)
    # print("Normal PB : {}".format(timeit.default_timer() - start_time))

    # start_time = timeit.default_timer()
    # # run frozen
    # run_altered(freeze_runs_dir)
    # print("Frozen : {}".format(timeit.default_timer() - start_time))

    # start_time = timeit.default_timer()
    # # run optimized
    # run_altered(optimize_runs_dir)
    # print("Optimized : {}".format(timeit.default_timer() - start_time))

    # start_time = timeit.default_timer()
    # # run 8 bit
    # run_altered(eight_bit_runs_dir)
    # print("8 Bit : {}".format(timeit.default_timer() - start_time))

if __name__ == '__main__':
    run()