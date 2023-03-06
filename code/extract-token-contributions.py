import sys
import pickle
import numpy as np
import tensorflow as tf
import lib
import lib.task.seq2seq.models.transformer_lrp as tr

from lib.ops.record_activations import recording_activations
from lib.layers.basic import dropout_scope
from lib.ops import record_activations as rec
from lib.layers.lrp import LRP

IN_FILE = sys.argv[1]
OUT_FILE = sys.argv[2]
VOC_PATH = sys.argv[3]
batch_size = int(sys.argv[4])
batch_idx = int(sys.argv[5])
MAX_SRC_LEN = int(sys.argv[6])
TGT_WINDOW_LEN = int(sys.argv[7])
CLIP_TGT_LEN = int(sys.argv[8])

start_line = batch_idx * batch_size
end_line = start_line + batch_size
sys.path.insert(0, '../') # insert your local path to the repo

inp_voc = pickle.load(open(VOC_PATH + 'src.voc', 'rb'))
out_voc = pickle.load(open(VOC_PATH + 'dst.voc', 'rb'))
input_lines = open(IN_FILE, 'r', encoding='utf-8').readlines()
path_to_ckpt = VOC_PATH + 'checkpoint/model-latest.npz' # specify the path to the model checkpoint

test_src, test_dst = [], []
for index, line in enumerate(input_lines):
    if index < start_line:
        continue
    elif index >= end_line:
        break
    src, tgt = line.strip().split("\t")[:2]
    test_src.append(src)
    test_dst.append(tgt)

tf.reset_default_graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99, allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.InteractiveSession(config=config)

hp = {
     "num_layers": 6,
     "num_heads": 8,
     "ff_size": 2048,
     "ffn_type": "conv_relu",
     "hid_size": 512,
     "emb_size": 512,
     "res_steps": "nlda", 
    
     "rescale_emb": True,
     "inp_emb_bias": True,
     "normalize_out": True,
     "share_emb": False,
     "replace": 0,
    
     "relu_dropout": 0.1,
     "res_dropout": 0.1,
     "attn_dropout": 0.1,
     "label_smoothing": 0.1,
    
     "translator": "ingraph",
     "beam_size": 4,
     "beam_spread": 3,
     "len_alpha": 0.6,
     "attn_beta": 0,
}

model = tr.Model('mod', inp_voc, out_voc, inference_mode='fast', **hp)
var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
lib.train.saveload.load(path_to_ckpt, var_list)

def get_topk_logits_selector(logits, k=3):
    """ takes logits[batch, nout, voc_size] and returns a mask with ones at k largest logits """
    topk_logit_indices = tf.nn.top_k(logits, k=k).indices
    indices = tf.stack([
        tf.range(tf.shape(logits)[0] * tf.shape(logits)[1] * k) // (tf.shape(logits)[1] * k),
        (tf.range(tf.shape(logits)[0] * tf.shape(logits)[1] * k) // k) % tf.shape(logits)[1],
        tf.reshape(topk_logit_indices, [-1])
    ], axis=1)
    ones = tf.ones(shape=(tf.shape(indices)[0],))
    return tf.scatter_nd(indices, ones, shape=tf.shape(logits))

def compute_LRP(ph):
    target_position = tf.placeholder(tf.int32, [])
    with rec.recording_activations() as saved_activations, dropout_scope(False):
        rdo = model.encode_decode(ph, is_train=False)
        logits = model.loss._rdo_to_logits(rdo)
        out_mask = tf.one_hot(target_position, depth=tf.shape(logits)[1])[None, :, None]

        R_ = get_topk_logits_selector(logits, k=1) * out_mask
        R = model.loss._rdo_to_logits.relprop(R_)
        R = model.transformer.relprop_decode(R)
        
        R_out = tf.reduce_sum(abs(R['emb_out']), axis=-1)
        R_inp = tf.reduce_sum(abs(model.transformer.relprop_encode(R['enc_out'])), axis=-1)
    return target_position, R_out, R_inp


result = []
num = 0
for elem in zip(test_src, test_dst):
    num += 1
    inp_lrp = []
    out_lrp = []
    src_orig = elem[0].strip()
    dst_orig = elem[1].strip()
    src = ' '.join(src_orig.split()[:MAX_SRC_LEN])
    dst = ' '.join(dst_orig.split()[:TGT_WINDOW_LEN])
    dst_words = min(len(dst_orig.split()) + 1, CLIP_TGT_LEN)
    print(num, len(src.split()), dst_words)
    feed_dict = model.make_feed_dict(zip([src], [dst]))
    ph = lib.task.seq2seq.data.make_batch_placeholder(feed_dict)
    feed = {ph[key]: feed_dict[key] for key in feed_dict}
    target_position, R_out, R_inp = compute_LRP(ph)
    for token_pos in range(dst_words):
        if token_pos - TGT_WINDOW_LEN > 0:
            dst = ' '.join(dst_orig.split()[token_pos - TGT_WINDOW_LEN :token_pos])
            feed_dict = model.make_feed_dict(zip([src], [dst]))
            feed = {ph[key]: feed_dict[key] for key in feed_dict}
            feed[target_position] = TGT_WINDOW_LEN
        else:
            feed[target_position] = token_pos
        res_inp, res_out = sess.run((R_inp, R_out), feed)
        inp_lrp.append(res_inp[0])
        out_lrp.append(res_out[0])
    result.append({'src': src_orig, 'dst': dst_orig,
                   'inp_lrp': np.array(inp_lrp), 'out_lrp': np.array(out_lrp)
                  })

pickle.dump(result, open(OUT_FILE, 'wb'))