
from basic.graph_handler import GraphHandler
from basic.model import get_multi_gpu_models
from basic.main import set_dirs

from http import HTTPStatus
import http.server
import urllib.parse as urlparse
import json

from nltk import word_tokenize

import tensorflow as tf
import numpy as np
import os
import io
import shutil
from tqdm import tqdm
from pprint import pprint

SESS = None
MODEL = None
SHARED = None
CONFIG = None

def get_feed_dict(ctx, ques):
  N, M, JX, JQ, VW, VC, d, W = \
      CONFIG.batch_size, CONFIG.max_num_sents, CONFIG.max_sent_size, \
      CONFIG.max_ques_size, CONFIG.word_vocab_size, CONFIG.char_vocab_size, CONFIG.hidden_size, CONFIG.max_word_size
  feed_dict = {}

  x = np.zeros([N, M, JX], dtype='int32')
  cx = np.zeros([N, M, JX, W], dtype='int32')
  x_mask = np.zeros([N, M, JX], dtype='bool')
  q = np.zeros([N, JQ], dtype='int32')
  cq = np.zeros([N, JQ, W], dtype='int32')
  q_mask = np.zeros([N, JQ], dtype='bool')

  feed_dict[MODEL.x] = x
  feed_dict[MODEL.x_mask] = x_mask
  feed_dict[MODEL.cx] = cx
  feed_dict[MODEL.q] = q
  feed_dict[MODEL.cq] = cq
  feed_dict[MODEL.q_mask] = q_mask
  feed_dict[MODEL.is_train] = False
  if CONFIG.use_glove_for_unk:
      feed_dict[MODEL.new_emb_mat] = SHARED['new_emb_mat']

  def _get_word(word):
    d = SHARED['word2idx']
    if word.lower() in d:
      return d[word.lower()]
    if CONFIG.use_glove_for_unk:
        d2 = SHARED['new_word2idx']
        if word.lower() in d2:
          return d2[word.lower()] + len(d)
    return 1

  def _get_char(char):
    d = SHARED['char2idx']
    if char in d:
        return d[char]
    return 1

  for k, xijk in enumerate(ctx):
      each = _get_word(xijk)
      assert isinstance(each, int), each
      x[0, 0, k] = each
      x_mask[0, 0, k] = True

  for k, cxijk in enumerate(ctx):
      for l, cxijkl in enumerate(cxijk):
        cx[0, 0, k, l] = _get_char(cxijkl)

  for j, qij in enumerate(ques):
      q[0, j] = _get_word(qij)
      q_mask[0, j] = True

  for j, cqij in enumerate(ques):
      for k, cqijk in enumerate(cqij):
          cq[0, j, k] = _get_char(cqijk)

  return feed_dict

def get_best_span(yp, yp2):
  argmax_end_after = []
  max_end = 0
  argmax_end = len(yp) - 1
  for i in range(len(yp)):
    j = len(yp) - i - 1
    curr = yp2[j]
    if curr > max_end:
      max_end = curr
      argmax_end = j
    argmax_end_after.append(argmax_end)
  max_span = 0
  argmax_span = (0, 0)
  for i in range(len(yp)):
    j = len(yp) - i - 1
    curr = yp[i] * yp2[argmax_end_after[j]]
    if curr > max_span:
      max_span = curr
      argmax_span = (i, argmax_end_after[j])
  return argmax_span, max_span

class QAHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
  def generate_response_body(self, c, q):
    c = word_tokenize(c)
    q = word_tokenize(q)
    yp, yp2 = SESS.run([MODEL.yp, MODEL.yp2], feed_dict=get_feed_dict(c, q))
    min_start = min(float(yp[0][1][0]), min([float(x) for x in yp[0][0][:len(c)]]))
    max_start = max(float(yp[0][1][0]), max([float(x) for x in yp[0][0][:len(c)]]))
    start_range = max_start - min_start
    min_end = min(float(yp2[0][1][0]), min([float(x) for x in yp2[0][0][:len(c)]]))
    max_end = max(float(yp2[0][1][0]), max([float(x) for x in yp2[0][0][:len(c)]]))
    end_range = max_end - min_end
    (start, end), score = get_best_span([float(x) for x in yp[0][0][:len(c)]], [float(x) for x in yp2[0][0][:len(c)]])
    no_ans_score = yp[0][1][0] * yp2[0][1][0]
    rows = []
    max_row_len = 15
    answer = []
    if score < no_ans_score:
      answer.append("No answer")
      for i, word in enumerate(c):
        if i % max_row_len == 0:
          curr_top_row, curr_mid_row, curr_bot_row = [], [], []
          rows.append(curr_top_row)
          rows.append(curr_mid_row)
          rows.append(curr_bot_row)
        curr_top_row.append("<td class=\"outAnswer\">%s</td>" % word)
        score_lerp = (float(yp[0][0][i]) - min_start) / start_range
        non_green = int((1 - score_lerp)*255)
        curr_mid_row.append("<td style=\"background-color: rgb(%d,255,%d)\">%.2f</td>" % (non_green, non_green, score_lerp))

        score_lerp = (float(yp2[0][0][i]) - min_end) / end_range
        non_red = int((1 - score_lerp)*255)
        curr_bot_row.append("<td style=\"background-color: rgb(255,%d,%d)\">%.2f</td>" % (non_red, non_red, score_lerp))
      rows.append(["<td class=\"inAnswer\">No Answer</td>"])
      score_lerp = (float(yp[0][1][0]) - min_start) / start_range
      rows.append(["<td style=\"background-color: rgb(197,71,255)\">%.2f</td>" % score_lerp])
      score_lerp = (float(yp2[0][1][0]) - min_end) / end_range
      rows.append(["<td style=\"background-color: rgb(197,71,255)\">%.2f</td>" % score_lerp])
    else:
      for i, word in enumerate(c):
        if i % max_row_len == 0:
          curr_top_row, curr_mid_row, curr_bot_row = [], [], []
          rows.append(curr_top_row)
          rows.append(curr_mid_row)
          rows.append(curr_bot_row)
        if i >= start and i <= end:
          answer.append(word)
          curr_top_row.append("<td class=\"inAnswer\">%s</td>" % word)
        else:
          curr_top_row.append("<td class=\"outAnswer\">%s</td>" % word)
        score_lerp = (float(yp[0][0][i]) - min_start) / start_range
        non_green = int((1 - score_lerp)*255)
        curr_mid_row.append("<td style=\"background-color: rgb(%d,255,%d)\">%.2f</td>" % (non_green, non_green, score_lerp))
        score_lerp = (float(yp2[0][0][i]) - min_end) / end_range
        non_red = int((1 - score_lerp)*255)
        curr_bot_row.append("<td style=\"background-color: rgb(255,%d,%d)\">%.2f</td>" % (non_red, non_red, score_lerp))
    print("Responding with answer: %s" % ' '.join(answer))
    for i in range(len(rows)//3):
      three = rows[i*3:(i+1)*3]
      if len(three[0]) > 0:
        yield "<table>"
        for row in three:
          yield "<tr>"
          for cell in row:
            yield cell
          yield "</tr>"
        yield "</table>"

  def do_GET(self):
    if self.path.startswith('/ask?'):
      self.send_response(HTTPStatus.OK)
      self.send_header("Content-type", "text/html")
      params = urlparse.parse_qs(self.path[5:])
      c = params['c'][0]
      q = params['q'][0]
      b = io.BytesIO()
      for item in self.generate_response_body(c, q):
        b.write(item.encode())
      b.seek(0)
      self.send_header("Content-Length", str(b.getbuffer().nbytes))
      self.send_header("Access-Control-Allow-Origin", '*')
      self.end_headers()
      shutil.copyfileobj(b, self.wfile)
    else:
      return super(QAHttpRequestHandler, self).do_GET()

flags = tf.app.flags

# Names and directories
flags.DEFINE_string("model_name", "basic", "Model name [basic]")
flags.DEFINE_string("data_dir", "data/squad", "Data dir [data/squad]")
flags.DEFINE_string("run_id", "0", "Run ID [0]")
flags.DEFINE_string("out_base_dir", "out", "out base dir [out]")
flags.DEFINE_string("forward_name", "single", "Forward name [single]")
flags.DEFINE_string("answer_path", "", "Answer path []")
flags.DEFINE_string("eval_path", "", "Eval path []")
flags.DEFINE_string("load_path", "", "Load path []")
flags.DEFINE_string("shared_path", "", "Shared path []")

# Device placement
flags.DEFINE_string("device", "/cpu:0", "default device for summing gradients. [/cpu:0]")
flags.DEFINE_string("device_type", "gpu", "device for computing gradients (parallelization). cpu | gpu [gpu]")
flags.DEFINE_integer("num_gpus", 1, "num of gpus or cpus for computing gradients [1]")

# Essential training and test options
flags.DEFINE_string("mode", "test", "train | dev | test | forward [test]")
flags.DEFINE_boolean("load", True, "load saved data? [True]")
flags.DEFINE_bool("single", False, "supervise only the answer sentence? [False]")
flags.DEFINE_boolean("debug", False, "Debugging mode? [False]")
flags.DEFINE_bool('load_ema', True, "load exponential average of variables when testing?  [True]")
flags.DEFINE_bool("eval", True, "eval? [True]")

# Training / test parameters
flags.DEFINE_integer("batch_size", 60, "Batch size [60]")
flags.DEFINE_integer("val_num_batches", 100, "validation num batches [100]")
flags.DEFINE_integer("test_num_batches", 0, "test num batches [0]")
flags.DEFINE_integer("num_epochs", 12, "Total number of epochs for training [12]")
flags.DEFINE_integer("num_steps", 20000, "Number of steps [20000]")
flags.DEFINE_integer("load_step", 0, "load step [0]")
flags.DEFINE_float("init_lr", 0.5, "Initial learning rate [0.5]")
flags.DEFINE_float("input_keep_prob", 0.8, "Input keep prob for the dropout of LSTM weights [0.8]")
flags.DEFINE_float("keep_prob", 0.8, "Keep prob for the dropout of Char-CNN weights [0.8]")
flags.DEFINE_float("wd", 0.0, "L2 weight decay for regularization [0.0]")
flags.DEFINE_integer("hidden_size", 100, "Hidden size [100]")
flags.DEFINE_integer("char_out_size", 100, "char-level word embedding size [100]")
flags.DEFINE_integer("char_emb_size", 8, "Char emb size [8]")
flags.DEFINE_string("out_channel_dims", "100", "Out channel dims of Char-CNN, separated by commas [100]")
flags.DEFINE_string("filter_heights", "5", "Filter heights of Char-CNN, separated by commas [5]")
flags.DEFINE_bool("finetune", False, "Finetune word embeddings? [False]")
flags.DEFINE_bool("highway", True, "Use highway? [True]")
flags.DEFINE_integer("highway_num_layers", 2, "highway num layers [2]")
flags.DEFINE_bool("share_cnn_weights", True, "Share Char-CNN weights [True]")
flags.DEFINE_bool("share_lstm_weights", True, "Share pre-processing (phrase-level) LSTM weights [True]")
flags.DEFINE_float("var_decay", 0.999, "Exponential moving average decay for variables [0.999]")

# Optimizations
flags.DEFINE_bool("cluster", False, "Cluster data for faster training [False]")
flags.DEFINE_bool("len_opt", False, "Length optimization? [False]")
flags.DEFINE_bool("cpu_opt", False, "CPU optimization? GPU computation can be slower [False]")

# Logging and saving options
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_integer("log_period", 100, "Log period [100]")
flags.DEFINE_integer("eval_period", 1000, "Eval period [1000]")
flags.DEFINE_integer("save_period", 1000, "Save Period [1000]")
flags.DEFINE_integer("max_to_keep", 20, "Max recent saves to keep [20]")
flags.DEFINE_bool("dump_eval", True, "dump eval? [True]")
flags.DEFINE_bool("dump_answer", True, "dump answer? [True]")
flags.DEFINE_bool("vis", False, "output visualization numbers? [False]")
flags.DEFINE_bool("dump_pickle", True, "Dump pickle instead of json? [True]")
flags.DEFINE_float("decay", 0.9, "Exponential moving average decay for logging values [0.9]")

# Thresholds for speed and less memory usage
flags.DEFINE_integer("word_count_th", 10, "word count th [100]")
flags.DEFINE_integer("char_count_th", 50, "char count th [500]")
flags.DEFINE_integer("sent_size_th", 400, "sent size th [64]")
flags.DEFINE_integer("num_sents_th", 8, "num sents th [8]")
flags.DEFINE_integer("ques_size_th", 30, "ques size th [32]")
flags.DEFINE_integer("word_size_th", 16, "word size th [16]")
flags.DEFINE_integer("para_size_th", 256, "para size th [256]")

# Advanced training options
flags.DEFINE_bool("squash", False, "squash the sentences into one? [False]")
flags.DEFINE_bool("swap_memory", True, "swap memory? [True]")
flags.DEFINE_string("data_filter", "max", "max | valid | semi [max]")
flags.DEFINE_bool("use_glove_for_unk", True, "use glove for unk [False]")
flags.DEFINE_bool("lower_word", True, "lowercase words")
flags.DEFINE_bool("known_if_glove", True, "consider as known if present in glove [False]")
flags.DEFINE_string("logit_func", "tri_linear", "logit func [tri_linear]")
flags.DEFINE_string("answer_func", "linear", "answer logit func [linear]")
flags.DEFINE_string("sh_logit_func", "tri_linear", "sh logit func [tri_linear]")

# Ablation options
flags.DEFINE_bool("use_char_emb", True, "use char emb? [True]")
flags.DEFINE_bool("use_word_emb", True, "use word embedding? [True]")
flags.DEFINE_bool("q2c_att", True, "question-to-context attention? [True]")
flags.DEFINE_bool("c2q_att", True, "context-to-question attention? [True]")
flags.DEFINE_bool("dynamic_att", False, "Dynamic attention [False]")

# Negative prediction
flags.DEFINE_bool("pred_negative", True, "predict whether question is answerable [True]")
flags.DEFINE_bool("force_answer", True, "force no negative answers [True]")

def get_word2vec(glove_dir):
    glove_corpus = "6B"
    glove_vec_size = 100
    glove_path = os.path.join(glove_dir, "glove.{}.{}d.txt".format(glove_corpus, glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            word2vec_dict[word.lower()] = vector

    return word2vec_dict

def main(_):
    global SESS
    global MODEL
    global CONFIG
    global SHARED

    home = os.path.expanduser("~")
    glove_dir = os.path.join(home, "data", "glove")
    word2vec_dict = get_word2vec(glove_dir)

    config = flags.FLAGS

    config.out_dir = os.path.join(config.out_base_dir, config.model_name, str(config.run_id).zfill(2))

    set_dirs(config)

    CONFIG = config
    config.max_num_sents = 1
    config.max_sent_size = 400
    config.max_ques_size = 64
    config.batch_size = 1
    config.max_word_size = 32
    config.max_para_size = 400

    shared_path = os.path.join(config.out_dir, "shared.json")
    with open(shared_path, 'r') as fh:
      SHARED = json.load(fh)
    SHARED['word2vec'] = word2vec_dict
    SHARED['lower_word2vec'] = word2vec_dict
    with tf.device(config.device):
      if config.use_glove_for_unk:
        # create new word2idx and word2vec
        word2vec_dict = SHARED['lower_word2vec'] if config.lower_word else SHARED['word2vec']
        new_word2idx_dict = {word: idx for idx, word in enumerate(word for word in word2vec_dict.keys() if word not in SHARED['word2idx'])}
        SHARED['new_word2idx'] = new_word2idx_dict
        offset = len(SHARED['word2idx'])
        word2vec_dict = SHARED['lower_word2vec'] if config.lower_word else SHARED['word2vec']
        new_word2idx_dict = SHARED['new_word2idx']
        idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
        # print("{}/{} unique words have corresponding glove vectors.".format(len(idx2vec_dict), len(word2idx_dict)))
        new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
        SHARED['new_emb_mat'] = new_emb_mat
        config.new_emb_mat = new_emb_mat
      config.char_vocab_size = len(SHARED['char2idx'])
      config.word_emb_size = len(next(iter(SHARED['word2vec'].values())))
      config.word_vocab_size = len(SHARED['word2idx'])

      pprint(config.__flags, indent=2)
      models = get_multi_gpu_models(config)
      model = models[0]
      graph_handler = GraphHandler(config, model)  # controls all tensors and variables in the graph, including loading /saving

      sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
      graph_handler.initialize(sess)

      SESS = sess
      MODEL = model

      httpd = http.server.HTTPServer(('', 8000), QAHttpRequestHandler)
      print("Serving on port", httpd.server_port)
      httpd.serve_forever()

if __name__ == "__main__":
    tf.app.run()