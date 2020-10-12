import random
from model import *

def multi():
    input_dims = 1000
    output_dim = 64

    params = dict()

    params['image_hidden'] = [4, 8, 16]
    params['text_hidden'] = [64, 128, 256]
    params['image_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params['text_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params['factor_learning_rate'] = [0.0003, 0.0005, 0.001, 0.003]
    params['learning_rate'] = [0.0003, 0.0005, 0.001, 0.003]
    params['rank'] = [1, 4, 8, 16]
    params['batch_size'] = [4, 8, 16, 32, 64, 128]
    params['weight_decay'] = [0, 0.001, 0.002, 0.01]

    ihid = random.choice(params['image_hidden'])
    thid = random.choice(params['text_hidden'])
    thid_2 = thid // 2
    idr = random.choice(params['image_dropout'])
    tdr = random.choice(params['text_dropout'])
    factor_lr = random.choice(params['factor_learning_rate'])
    lr = random.choice(params['learning_rate'])
    r = random.choice(params['rank'])
    batch_sz = random.choice(params['batch_size'])
    decay = random.choice(params['weight_decay'])

    model = LMF(input_dims, (ihid, thid), thid_2, (idr, tdr, 0.5), output_dim, r)
