#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:Wang Yan
@ide:PyCharm
@time:2019/7/17 21:47
"""
from load_data import *
from model import *
import matplotlib.pyplot as plt


# 测试检查点
def eval():
    N_CLASSES = 2
    IMG_SIZE = 208
    BATCH_SIZE = 1
    CAPACITY = 200
    MAX_STEP = 12500

    test_dir = 'data/test'
    logs_dir = 'logs_1'     # 检查点目录

    sess = tf.Session()

    train_list = get_all_files(test_dir, is_random=True)
    image_train_batch, label_train_batch = get_batch(train_list, IMG_SIZE, BATCH_SIZE, CAPACITY, True)
    train_logits = inference(image_train_batch, N_CLASSES)
    train_logits = tf.nn.softmax(train_logits)  # 用softmax转化为百分比数值

    # 载入检查点
    saver = tf.train.Saver()
    print('\n载入检查点...')
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功，global_step = %s\n' % global_step)
    else:
        print('没有找到检查点')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            image, prediction = sess.run([image_train_batch, train_logits])
            print(prediction)
            max_index = np.argmax(prediction)
            if max_index == 0:
                label = '%.2f%% is a cat.' % (prediction[0][0] * 100)
            else:
                label = '%.2f%% is a dog.' % (prediction[0][1] * 100)
            plt.imshow(image[0])
            plt.title(label)
            plt.show()

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()


if __name__ == '__main__':
    eval()
