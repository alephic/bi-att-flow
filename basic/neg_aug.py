import json
import random

data_path = 'data/squad/'

def run():
  random.seed(42)
  base_shared_train = json.load(open(data_path+'shared_train.json'))
  base_shared_dev = json.load(open(data_path+'shared_dev.json'))
  base_shared_test = json.load(open(data_path+'shared_test.json'))
  base_data_train = json.load(open(data_path+'data_train.json'))
  base_data_dev = json.load(open(data_path+'data_dev.json'))
  base_data_test = json.load(open(data_path+'data_test.json'))
  json.dump(add_neg_examples(base_data_train, base_shared_train), open(data_path+'data_neg_train.json', mode='w'))
  json.dump(add_neg_examples(base_data_dev, base_shared_dev), open(data_path+'data_neg_dev.json', mode='w'))
  json.dump(add_neg_examples(base_data_test, base_shared_test), open(data_path+'data_neg_test.json', mode='w'))

def add_neg_examples(data, shared):
  keys = ['*p', '*x', '*cx', 'q', 'cq', 'y', 'cy', 'answerss', 'ids', 'idxs']
  new_examples = {k:[] for k in keys}
  for idx in data['idxs']:
    rp, rx, rcx, q, cq, y, cy, answerss, ids, idxs = [data[k][idx] for k in keys]
    sents = shared['x'][rx[0]][rx[1]]
    new_y = [[[len(sents), 0], [len(sents), 1]]]
    new_rx = [random.randint(0, len(shared['x']) - 1)]
    new_rx.append(random.randint(0, len(shared['x'][new_rx[0]]) - 1))
    while new_rx == rx:
      # Make sure same context isn't chosen
      new_rx = [random.randint(0, len(shared['x']) - 1)]
      new_rx.append(random.randint(0, len(shared['x'][new_rx[0]]) - 1))
    
    data['*p'].append(new_rx)
    data['*x'].append(new_rx)
    data['*cx'].append(new_rx)
    data['q'].append(q)
    data['cq'].append(cq)
    data['y'].append(new_y)
    data['cy'].append([[0, 0]])
    data['answerss'].append([''])
    data['ids'].append('')
    data['idxs'].append(len(data['idxs']))
  return data

if __name__ == "__main__":
  run()