import numpy as np
import tensorflow as tf

from src.model import model
from src.utl.load_param import model_dir, pred_dir
from src.utl.utl import try_mkdir

# TF compatability:
Session = tf.compat.v1.Session

def load(model_name: str) -> (Session, dict):

    # TF compatability:
    ConfigProto = tf.compat.v1.ConfigProto
    all_variables = tf.compat.v1.global_variables
    Saver = tf.compat.v1.train.Saver

    init, net1_optim, net2_optim, all_optim, x, x_face_id, y_landmark, y_phoneme, y_lipS, y_maya_param, dropout, cost, \
    tensorboard_op, pred, clear_op, inc_op, avg, batch_size_placeholder, phase = model()

    # start tf graph
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = Session(config=config)
    max_to_keep = 20

    # thanks to https://github.com/tensorflow/tensorflow/issues/2768#issuecomment-225065522

    names_to_vars = {v.op.name.replace("bias", "biases").replace("kernel", "weights"): v for v in all_variables()}

    saver = Saver(max_to_keep=max_to_keep, var_list=names_to_vars)

    try_mkdir(pred_dir)

    # Test sess, load ckpt
    OLD_CHECKPOINT_FILE = model_dir + model_name + '/' + model_name +'.ckpt'

    print("Model loading: " + OLD_CHECKPOINT_FILE)
    saver.restore(sess, OLD_CHECKPOINT_FILE)
    print("Model loaded: " + model_dir + model_name)

    model_dict = {
        "init": init,
        "net1_optim": net1_optim,
        "net2_optim": net2_optim,
        "all_optim": all_optim,
        "x": x,
        "x_face_id": x_face_id,
        "y_landmark": y_landmark,
        "y_phoneme": y_phoneme,
        "y_lips": y_lipS,
        "y_maya_param": y_maya_param,
        "dropout": dropout,
        "cost": cost,
        "tensorboard_op": tensorboard_op,
        "pred": pred,
        "clear_op": clear_op,
        "inc_op": inc_op,
        "avg": avg,
        "batch_size_placeholder": batch_size_placeholder,
        "phase": phase,
    }
    return sess, model_dict
