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
from const import TRAIN_MOUSE_INDEX, PATCH_SIZE, MOUSE_N, RESULT_DIR, SAVE_TRAIN_IMG, MAX_EPOCH, SAVE_EVERY_N_EPOCH, TV_ALPHA


def main(settings: Settings):
    input_channels = 1
    output_channels = 1
    nx, ny, nz = PATCH_SIZE
    batch_size = settings.get_batch()

    all_mouse_idx = range(0,MOUSE_N)
    tr_mouse_idx = TRAIN_MOUSE_INDEX
    te_mouse_idx = [e for e in all_mouse_idx if e not in TRAIN_MOUSE_INDEX]

    # Load data
    train_list = []
    test_list = []
    for mouse_i in range(MOUSE_N):
        print('reading %d'%mouse_i)
        if mouse_i in tr_mouse_idx:
            train_list.append(Mouse(mouse_i,
                                    is_test=False,
                                    preprocess=True,
                                    img_num=settings.get_image_number(),
                                    prefix=settings.get_prefix()
                                    ))
        else:
            test_list.append(Mouse(mouse_i,
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

    # Test result variable definition
    tr_loss = []  # type: List[float]
    te_loss = []  # type: List[float]
    metrics_train = []  # tn, fn, fp, tp
    metrics_test = []  # tn, fn, fp, tp

    # Accuracy scores
    metrics = set_metrics(y_, y_predicted)

    saver = tf.train.Saver(max_to_keep=200)
    log_merge = tf.summary.merge_all()

    # Run part
    with tf.Session() as sess:  # launch the model in an interactive session
        sess.run(tf.global_variables_initializer())  # create an operation to initialize the variables we created
        sess.run(tf.local_variables_initializer())  # needed to compute tp, tn, fp, fn
        #log_writer = tf.summary.FileWriter('./log', sess.graph, flush_secs=60)
        for epoch_i in range(1,MAX_EPOCH):
            
            ############### Training for one epoch ###############
            ntrstep = 1
            for train_batch in batch_generator(train_list, batch_size): #(no. of traiing steps =  no. of raining patches)
 
                feed_dict = {
                    x: train_batch.get_original_images_in_batch(),
                    y_: train_batch.get_ground_truths_in_batch(),
                    is_train: True
                }

                # Training step
                t = time.time()
                _, total_loss_p, bce_loss_p, l1wreg_loss_p, log_summary = sess.run([train_step, total_loss, bce_loss, l1wreg_loss, log_merge], feed_dict)
                elapsed = time.time() - t

                # Printing diagnostics and cross-entropy loss
                print('[epoch: %003d] [trainstep: %003d] [iter_time(s): %1.3f] [loss: %1.5f] [l1reg: %1.5f]' % (epoch_i, ntrstep, elapsed, bce_loss_p, l1wreg_loss_p))
                
                # Save log
                #log_writer.add_summary(log_summary, ntrstep)
                ntrstep = ntrstep + 1

            ############### Saving diagnostics ###############
            if (epoch_i) % SAVE_EVERY_N_EPOCH == 0:

                # Print diagnostic message
                print('*** Testing ***')
                
                # Initialize metric containers
                metrics_train_tmp = []
                metrics_test_tmp = []
                tr_loss_tmp = []
                te_loss_tmp = []

                # Timer
                t = time.time()

                if SAVE_TRAIN_IMG == 1:
                    # Classify full FoV training data
                    for trainmouse in train_list:
                        y_tr_cube_predicted = np.empty(trainmouse.shape, dtype='uint8')
                        y_tr_cube_possibility = np.empty(trainmouse.shape, dtype='float32')

                        for train_batch in batch_generator([trainmouse], batch_size):
                            feed_dict = {
                                x: train_batch.get_original_images_in_batch(),
                                y_: train_batch.get_ground_truths_in_batch()
                            }

                            y_pos_p, y_pred_p, total_loss_p, metrics_p = sess.run(
                                (y_sigmoid, y_predicted, total_loss, metrics), feed_dict)

                            metrics_train_tmp.append(metrics_p)
                            tr_loss_tmp.append(total_loss_p)
                            
                            # put data into full FoV volume for testmouse
                            for patch_i in range(batch_size):
                                y_tr_cube_predicted[train_batch.get_index_list()[patch_i]] = y_pred_p[patch_i, ..., 0]
                                y_tr_cube_possibility[train_batch.get_index_list()[patch_i]] = y_pos_p[patch_i, ..., 0]

                        # Save classification result on training data
                        img_save_dir = result_dir + "img/epoch%d/tr_mouse%d" % (epoch_i, trainmouse.mousenum)
                        try:
                            os.makedirs(img_save_dir)
                        except FileExistsError:
                            pass
                        save_3d_tiff(img_save_dir,
                                     {'predicted': y_tr_cube_predicted,
                                      'possibility': y_tr_cube_possibility})
                    # Save metrics
                    metrics_train_tmp_reduced = np.sum(metrics_train_tmp, axis=0)
                    metrics_train.append(metrics_train_tmp_reduced)
                    tr_loss_tmp_reduced = np.mean(tr_loss_tmp) 
                    tr_loss.append(tr_loss_tmp_reduced)
                else:
                    total_loss_p, metrics_p = sess.run((total_loss, metrics), feed_dict)
                    tr_loss.append(total_loss_p)
                    metrics_train.append(metrics_p)
                    
                
                # Classify full FoV testing data (one test mouse at a time)
                for testmouse in test_list:
                    y_te_cube_predicted = np.empty(testmouse.shape, dtype='uint8')
                    y_te_cube_possibility = np.empty(testmouse.shape, dtype='float32')

                    for test_batch in batch_generator([testmouse], batch_size):
                        feed_dict = {
                            x: test_batch.get_original_images_in_batch(),
                            y_: test_batch.get_ground_truths_in_batch(),
                            is_train: False
                        }

                        y_pos_p, y_pred_p, total_loss_p, metrics_p = sess.run(
                            (y_sigmoid, y_predicted, total_loss, metrics), feed_dict)

                        if testmouse.mousenum != 6: # do NOT compute metrics or loss for mouse6 (new mouse w/o gtruth data)
                           metrics_test_tmp.append(metrics_p)
                           te_loss_tmp.append(total_loss_p)
                        
                        # put data into full FoV volume for testmouse
                        for patch_i in range(batch_size):
                            y_te_cube_predicted[test_batch.get_index_list()[patch_i]] = y_pred_p[patch_i, ..., 0]
                            y_te_cube_possibility[test_batch.get_index_list()[patch_i]] = y_pos_p[patch_i, ..., 0]

                    # Save classification result on test data
                    img_save_dir = result_dir + "img/epoch%d/te_mouse%d" % (epoch_i, testmouse.mousenum)
                    try:
                        os.makedirs(img_save_dir)
                    except FileExistsError:
                        pass
                    save_3d_tiff(img_save_dir,
                                 {'predicted': y_te_cube_predicted,
                                  'possibility': y_te_cube_possibility})

                elapsed_t = time.time() - t

                # Computing testing metrics | 0-3: (tn, fn, fp, tp)
                metrics_test_tmp_reduced = np.sum(metrics_test_tmp, axis=0)
                metrics_test.append(metrics_test_tmp_reduced)

                # Compute test loss
                te_loss_tmp_reduced = np.mean(te_loss_tmp)  # type: float
                te_loss.append(te_loss_tmp_reduced)
                

                # Save accuracy metrics | tp, tn, fp, fn
                metrics_fn = '%smetrics_train_test_tp_tn_fp_fn' % result_dir
                np.savez(metrics_fn, np.transpose(np.array(metrics_train))[[3, 0, 2, 1]],
                         np.transpose(np.array(metrics_test))[[3, 0, 2, 1]])

                # Save Loss value
                losses_fn = '%sloss_tr_te' % result_dir
                np.savez(losses_fn, np.asarray(tr_loss, dtype=np.float32), np.asarray(te_loss, dtype=np.float32))

                # Save model
                saver.save(sess, "./chkpt/my-model", global_step=epoch_i)

                # print testing diagnostics
                print('[diagnostics time: %1.5f][test loss: %1.5f] \n' % (elapsed_t, te_loss_tmp_reduced,))
            
            
def set_metrics(label, predicted):
    with tf.name_scope("metrics_tn_fn_fp_tp"):
        label = tf.cast(label, tf.int32)
        predicted = tf.cast(predicted, tf.int32)
        return tf.bincount(label + 2 * predicted)


def set_metrics2(label, predicted):
    c_matrix = cmatrix(label.ravel(),predicted.ravel(),labels=[1,0])
    c = c_matrix.ravel() #tp,fn,fp,tn
    ct = np.transpose(np.array(c))[[0, 3, 2, 1]] #reorder to: tp,tn,fp,fn
    return ct

if __name__ == '__main__':
    opt = Settings()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print('Tensorflow version')
    print(tf.__version__)
    main(opt)
