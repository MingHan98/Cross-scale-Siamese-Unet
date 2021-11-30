import tensorflow as tf
import sys
import random
import time
import labelreg.helpers as helper
import labelreg.networks as network
import labelreg.utils as util
import labelreg.losses as loss
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.compat.v1.disable_eager_execution()

# from tensorflow import ConfigProto
# from tensorflow import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# 0 - get configs
config = helper.ConfigParser(sys.argv, 'training')

# 1 - data
reader_moving_image, reader_fixed_image, reader_moving_label, reader_fixed_label = helper.get_data_readers(
    config['Data']['dir_moving_image'],
    config['Data']['dir_fixed_image'],
    config['Data']['dir_moving_label'],
    config['Data']['dir_fixed_label'])

# 2 - graph

ph_moving_image = tf.compat.v1.placeholder(tf.float32,
                                           [config['Train']['minibatch_size']] + reader_moving_image.data_shape + [1])  # placeholder
ph_fixed_image = tf.compat.v1.placeholder(tf.float32,
                                          [config['Train']['minibatch_size']] + reader_fixed_image.data_shape + [1])
ph_moving_affine = tf.compat.v1.placeholder(tf.float32,
                                            [config['Train']['minibatch_size']] + [1, 12])
ph_fixed_affine = tf.compat.v1.placeholder(tf.float32, [config['Train']['minibatch_size']] + [1, 12])
input_moving_image = util.warp_image_affine(ph_moving_image, ph_moving_affine)  # data augmentation
input_fixed_image = util.warp_image_affine(ph_fixed_image, ph_fixed_affine)  # data augmentation

ph_moving_label = tf.compat.v1.placeholder(tf.float32,
                                           [config['Train']['minibatch_size']] + reader_moving_image.data_shape + [1])
ph_fixed_label = tf.compat.v1.placeholder(tf.float32,
                                          [config['Train']['minibatch_size']] + reader_fixed_image.data_shape + [1])
input_moving_label = util.warp_image_affine(ph_moving_label, ph_moving_affine)  # data augmentation
input_fixed_label = util.warp_image_affine(ph_fixed_label, ph_fixed_affine)  # data augmentation


# all_image = tf.concat([input_moving_image, input_fixed_image, input_moving_label, input_fixed_label], axis=0)
# input_moving_image, input_fixed_image, input_moving_label, input_fixed_label = util.flip_random_image(all_image)
# predicting ddf


# input_moving_image = tf.concat([input_moving_image, input_moving_label], axis=4)
# input_fixed_image = tf.concat([input_fixed_image, input_moving_label], axis=4)

reg_net = network.build_network(network_type=config['Network']['network_type'],
                                minibatch_size=config['Train']['minibatch_size'],
                                image_moving=input_moving_image,
                                image_fixed=input_fixed_image)

# loss

# warped_moving_label = reg_net.warp_image(input_moving_label)  # warp the moving label with the predicted ddf


# warped_moving_label = reg_net.warp_image(input_moving_label)  # warp the moving label with the predicted ddf

warped_moving_label = reg_net.warp_image(input_moving_label)

# loss_similarity, loss_regulariser, loss_distance = loss.build_loss_2(similarity_type=config['Loss']['similarity_type'],
#                                                                      similarity_scales=config['Loss']['similarity_scales'],
#                                                                      regulariser_type=config['Loss']['regulariser_type'],
#                                                                      regulariser_weight=config['Loss']['regulariser_weight'],
#                                                                      label_moving=warped_moving_label,
#                                                                      label_fixed=input_fixed_label,
#                                                                      network_type=config['Network']['network_type'],
#                                                                      ddf=reg_net.ddf)

# warped_moving_image = reg_net.warp_image(input_moving_image)
# loss_similarity, loss_regulariser, loss_slcc = loss.build_loss_1(similarity_type=config['Loss']['similarity_type'],
#                                                                  similarity_scales=config['Loss']['similarity_scales'],
#                                                                  regulariser_type=config['Loss']['regulariser_type'],
#                                                                  regulariser_weight=config['Loss']['regulariser_weight'],
#                                                                  label_moving=warped_moving_label,
#                                                                  label_fixed=input_fixed_label,
#                                                                  network_type=config['Network']['network_type'],
#                                                                  ddf=reg_net.ddf,
#                                                                  image_warped=warped_moving_image,
#                                                                  image_fixed=input_fixed_image)
#
loss_similarity, loss_regulariser = loss.build_loss(similarity_type=config['Loss']['similarity_type'],
                                                    similarity_scales=config['Loss']['similarity_scales'],
                                                    regulariser_type=config['Loss']['regulariser_type'],
                                                    regulariser_weight=config['Loss']['regulariser_weight'],
                                                    label_moving=warped_moving_label,
                                                    label_fixed=input_fixed_label,
                                                    network_type=config['Network']['network_type'],
                                                    ddf=reg_net.ddf)
# Loss_summary = loss_similarity + loss_regulariser + loss_distance

Loss_summary = loss_similarity + loss_regulariser

Loss = Loss_summary

# trainable_vars = tf.trainable_variables()
# accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in trainable_vars]
# zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
# update_ops = tf..v1.get_collection(tf..v1.GraphKeys.UPDATE_OPS)
# accumlation_steps = 4
# with tf.control_dependencies(update_ops):
#     global_steps = tf.Variable(0, name='global_step', trainable=False)
#     optimizer = tf..v1.train.AdamOptimizer(config['Train']['learning_rate'])
#     grads = optimizer.compute_gradients(Loss, trainable_vars)
#     accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(grads)]
#     train_op = optimizer.apply_gradients([(accum_ops[i] / accumlation_steps, gv[1]) for i, gv in enumerate(grads)], \
#                                          global_step=global_steps)

train_op = tf.compat.v1.train.AdamOptimizer(config['Train']['learning_rate']).minimize(Loss_summary)
# utility nodes - for information only
dice = util.compute_binary_dice(warped_moving_label, input_fixed_label)
dist = util.compute_centroid_distance(warped_moving_label, input_fixed_label)
tf.compat.v1.summary.scalar('Loss_summary', Loss_summary)
tf.compat.v1.summary.scalar('Loss', loss_similarity)
tf.compat.v1.summary.scalar('Loss_regulariser', loss_regulariser)
# tf.compat.v1.summary.scalar('Loss_distance', loss_distance)
# tf..v1.summary.scalar('Loss_slcc', loss_slcc)

merged = tf.compat.v1.summary.merge_all()

# 3 - training

num_minibatch = int(reader_moving_label.num_data / config['Train']['minibatch_size'])
train_indices = [i for i in range(reader_moving_label.num_data)]

saver = tf.compat.v1.train.Saver(max_to_keep=1)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

writer = tf.compat.v1.summary.FileWriter('/home/amax/SMH/label-reg-master', sess.graph)
best_loss = 1000000
print('----- Training Start -----')
for step in range(config['Train']['total_iterations']):

    if step in range(0, config['Train']['total_iterations'], num_minibatch):

        random.shuffle(train_indices)

    minibatch_idx = step % num_minibatch

    case_indices = train_indices[
                   minibatch_idx * config['Train']['minibatch_size']:(minibatch_idx + 1) * config['Train'][
                       'minibatch_size']]
    label_indices = [random.randrange(reader_moving_label.num_labels[i]) for i in case_indices]

    trainFeed = {ph_moving_image: reader_moving_image.get_data_moving_jdm(case_indices),
                 ph_fixed_image: reader_fixed_image.get_data_fixed(case_indices),
                 ph_moving_label: reader_moving_label.get_data_moving_jdm(case_indices, label_indices),
                 ph_fixed_label: reader_fixed_label.get_data_fixed(case_indices, label_indices),
                 ph_moving_affine: helper.random_transform_generator(config['Train']['minibatch_size']),
                 ph_fixed_affine: helper.random_transform_generator(config['Train']['minibatch_size'])}

    # sess.run(accum_ops, feed_dict=trainFeed)
    #
    # if ((step + 1) % accumlation_steps) == 0:
    #     sess.run(train_op, feed_dict=trainFeed)
    #     sess.run(zero_ops)


    sess.run(train_op, feed_dict=trainFeed)
    if step in range(0, config['Train']['total_iterations'], config['Train']['freq_info_print']):
        current_time = time.asctime(time.localtime())

        # loss_similarity_train, loss_regulariser_train, loss_distance_train, dice_train, dist_train = sess.run(
        #     [loss_similarity,
        #      loss_regulariser,
        #      loss_distance,
        #      dice,
        #      dist],
        #     feed_dict=trainFeed)
        # Loss_all = loss_similarity_train + loss_regulariser_train + loss_distance_train
        # result = sess.run(merged, feed_dict={Loss_summary: Loss_all, loss_similarity: loss_similarity_train,
        # loss_regulariser: loss_regulariser_train, loss_distance: loss_distance_train})
        # writer.add_summary(result, step)
        # print('Step %d [%s]: Loss=%f (similarity=%f, regulariser=%f， distance=%f)' %
        #       (step,
        #        current_time,
        #        loss_similarity_train + loss_regulariser_train + loss_distance_train,
        #        loss_similarity_train,
        #        loss_regulariser_train,
        #        loss_distance_train))


        loss_similarity_train, loss_regulariser_train,  dice_train, dist_train = sess.run(
            [loss_similarity,
             loss_regulariser,
             dice,
             dist],
            feed_dict=trainFeed)
        Loss_all = loss_similarity_train + loss_regulariser_train
        result = sess.run(merged, feed_dict={Loss_summary: Loss_all, loss_similarity: loss_similarity_train,
                                             loss_regulariser: loss_regulariser_train})  # merged也是需要run的
        writer.add_summary(result, step)
        print('Step %d [%s]: Loss=%f (similarity=%f, regulariser=%f)' %
              (step,
               current_time,
               loss_similarity_train + loss_regulariser_train,
               loss_similarity_train,
               loss_regulariser_train))
        this_loss = loss_similarity_train + loss_regulariser_train

        print('Dice: %s' % dice_train)
        print('Distance: %s' % dist_train)
        print('Images: %s' % case_indices)
    if step in range(0, config['Train']['total_iterations'], config['Train']['freq_model_save']):
        save_path = saver.save(sess, config['Train']['file_model_save'], write_meta_graph=False)
        print("Model saved in: %s" % save_path)
sess.close()
print("----- Training Over -----")
