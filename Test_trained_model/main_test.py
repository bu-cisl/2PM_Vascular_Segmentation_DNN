# V-net for biological image segmentation
from typing import List

import tensorflow as tf
import numpy as np
# from VNetOriginal import v_net
from sklearn.metrics import jaccard_similarity_score as j_score
from sklearn.metrics import precision_score as p_score
from sklearn.metrics import confusion_matrix as cmatrix
# from sklearn.metrics import recall_score as r_score
import os
import time
from data_io import batch_generator, Mouse, save_3d_tiff
from settings import Settings
from const import TEST_MOUSE_INDEX, PATCH_SIZE, MOUSE_N, SAVE_TRAIN_IMG, MAX_EPOCH, SAVE_EVERY_N_EPOCH, TV_ALPHA, RESULT_DIR
from os import listdir
from os.path import isfile, join


def main(settings: Settings):
    input_channels = 1
    output_channels = 1
    nx, ny, nz = PATCH_SIZE
    batch_size = settings.get_batch()

    te_mouse_idx = TEST_MOUSE_INDEX

    # Load data
    datapath = './test_data'
    matfiles = []
    # Filenames with full path
    for f in listdir(datapath):
        if isfile(join(datapath, f)) and f.endswith('.mat'):
            matfiles.append(join(datapath, f))

    # Filenames without full path
    matfile_names_only = []
    for f in listdir(datapath):
        if isfile(join(datapath, f)) and f.endswith('.mat'):
            matfile_names_only.append(f)

    test_list = []
    for mouse_n in matfiles:
        print('reading data %s for testing'%mouse_n)
        test_list.append(Mouse(mouse_n,
                               is_test=True,
                               preprocess=True,
                               img_num=1,  # test original image only
                               prefix=settings.get_prefix()
                               ))

   # Make folders to save diagnostics
    result_dir = RESULT_DIR
    try:
        os.makedirs(result_dir)
    except FileExistsError:
                pass 

    # Create the model
    with tf.name_scope('external_data'):
        # change input data type to float32
        x = tf.placeholder(dtype=tf.float32, shape=(batch_size, nx, ny, nz, input_channels), name="x")
        y_ = tf.placeholder(dtype=tf.float32, shape=(batch_size, nx, ny, nz, output_channels), name="y_")
        is_train = tf.placeholder(dtype=tf.bool, name="is_train")

    # Build the graph for the deep net. WARNING: Output channels change to 1
    y_conv = settings.get_net()(x, input_channels, output_channels, is_train)
    y_sigmoid = tf.sigmoid(y_conv)
    with tf.name_scope('Predict_result'):
        y_predicted = tf.to_int32(y_conv > 0, name="result_img")

    # Loss: cross entropy
    bce_loss = settings.get_loss()(y_conv, y_, alpha=TV_ALPHA)
    l1wreg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    total_loss = bce_loss + l1wreg_loss

    tf.summary.scalar("loss", total_loss)

    # Optimizer: Adam
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_step = settings.get_optimizer().minimize(total_loss, name="train_step")  # train_step is an operation
    train_step = tf.group([train_step, update_ops])


    saver = tf.train.Saver(max_to_keep=200)
    log_merge = tf.summary.merge_all()

    # Run part
    with tf.Session() as sess:  # launch the model in an interactive session
        sess.run(tf.global_variables_initializer())  # create an operation to initialize the variables we created
        sess.run(tf.local_variables_initializer())  # needed to compute tp, tn, fp, fn

        # Load saved model
        saver.restore(sess, "./chkpt_saved/my-model-100")

        # Print diagnostic message
        print('*** Testing ***')
        

        # Classify full FoV testing data (one test mouse at a time)
        count = 0
        for testmouse in test_list:
            y_te_cube_predicted = np.empty(testmouse.shape, dtype='uint8')

            for test_batch in batch_generator([testmouse], batch_size):
                feed_dict = {
                    x: test_batch.get_original_images_in_batch(),
                    is_train: False
                }

                y_pred_p = sess.run((y_predicted), feed_dict)

                
                # put data into full FoV volume for testmouse
                for patch_i in range(batch_size):
                    y_te_cube_predicted[test_batch.get_index_list()[patch_i]] = y_pred_p[patch_i, ..., 0]

            # Save classification result on test data
            img_save_dir = result_dir
            fn = os.path.splitext(matfile_names_only[count])[0]
            save_3d_tiff(result_dir, {fn: y_te_cube_predicted})
            count = count + 1
            

if __name__ == '__main__':
    opt = Settings()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print('Tensorflow version')
    print(tf.__version__)
    main(opt)
