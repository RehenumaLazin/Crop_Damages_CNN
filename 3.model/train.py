# Refactored Training Script for Histogram-based CNN Model

import os
import numpy as np
import logging
import tensorflow.compat.v1 as tf

from nnet_for_hist import Config, NeuralModel

tf.disable_v2_behavior()

def load_dataset(config):
    filename = 'histogram_Corn_MN_MO_metForcing_May_June_without_LW_HAND.npz'
    data = np.load(os.path.join(config.load_path, filename))
    return {
        'images': data['output_image'],
        'yields': data['output_area'],
        'years': data['output_year'],
        'locations': data['output_locations'],
        'indices': data['output_index']
    }

def build_val_mask(locations, validation_keys, years):
    mask_val = np.zeros(len(locations), dtype=bool)
    mask_train = np.zeros(len(locations), dtype=bool)
    for i, (loc, year) in enumerate(zip(locations, years)):
        loc_str = f"{int(loc[0])}_{int(loc[1])}"
        if loc_str in validation_keys and year not in [2012, 2017]:
            mask_val[i] = True
        elif loc_str not in validation_keys and year not in [2012, 2017]:
            mask_train[i] = True
    return mask_train, mask_val

def inverse_log_transform(y, k, c, max_val):
    return np.power(10, (y - c) / k) - max_val

def evaluate_model(sess, model, images, yields, k, c, max_val, config, time):
    pred_all, real_all = [], []
    for i in range(0, len(images), config.B):
        batch_x = images[i:i+config.B, :, time:time+config.H, :]
        batch_y = yields[i:i+config.B]
        pred = sess.run(model.logits, feed_dict={model.x: batch_x, model.y: batch_y, model.keep_prob: 1})
        pred = inverse_log_transform(pred, k, c, max_val)
        real = inverse_log_transform(batch_y, k, c, max_val)
        pred_all.append(pred)
        real_all.append(real)
    pred_all = np.concatenate(pred_all)
    real_all = np.concatenate(real_all)
    rmse = np.sqrt(np.mean((pred_all - real_all) ** 2))
    return rmse, pred_all, real_all

def train_and_evaluate(config, data, train_mask, val_mask, loop_id, start_time):
    image_all = data['images']
    yield_all = data['yields']
    logging.basicConfig(filename=os.path.join(config.save_path, f'train_loop_{loop_id}.log'), level=logging.DEBUG)

    # Load transformation params
    k = np.load('k.npy')
    c = np.load('c.npy')
    max_val = np.load('MaxValue.npy')

    # Build model
    tf.reset_default_graph()
    with tf.Graph().as_default():
        model = NeuralModel(config, 'net')
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.22))) as sess:
            sess.run(tf.global_variables_initializer())
            best_rmse = float('inf')
            for i in range(config.train_step):
                idx = np.random.choice(np.where(train_mask)[0], size=config.B, replace=False)
                x_batch = image_all[idx, :, start_time:start_time+config.H, :]
                y_batch = yield_all[idx]
                feed = {model.x: x_batch, model.y: y_batch, model.lr: config.lr, model.keep_prob: config.keep_prob}
                _, loss = sess.run([model.train_op, model.loss_err], feed_dict=feed)
                if i % 500 == 0:
                    val_rmse, _, _ = evaluate_model(sess, model, image_all[val_mask], yield_all[val_mask], k, c, max_val, config, start_time) # For 
                    print(f"Step {i}, Validation RMSE: {val_rmse:.4f}")
                    if val_rmse < best_rmse:
                        best_rmse = val_rmse
                        saver.save(sess, os.path.join(config.save_path, f'model_loop_{loop_id}.ckpt'))

if __name__ == '__main__':
    config = Config()
    config.H = 20 #For May-June H=20, for Jul-Nov H=50
    data = load_dataset(config)
    all_locs = data['indices']

    for loop in range(4):
        start_time = 0 #For May-June start time =0, For Jul_nov start_time =21
        val_keys = np.load(f'k{loop+1}.npy').tolist()
        train_mask, val_mask = build_val_mask(all_locs, val_keys, data['years'])
        print(f"Loop {loop+1}: Train {np.sum(train_mask)}, Val {np.sum(val_mask)}")
        train_and_evaluate(config, data, train_mask, val_mask, loop+1, start_time)
