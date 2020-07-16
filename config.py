import configparser
cfg_parser = configparser.ConfigParser()
cfg_parser.read('hyper_param.ini')

dropout_rate = cfg_parser.getfloat('default', 'dropout_rate', fallback=0.25)
learning_rate = cfg_parser.getfloat('default', 'learning_rate', fallback=0.0001)
batch_size = cfg_parser.getint('default', 'batch_size', fallback=32)
checkpoint_dir = cfg_parser.get('default', 'checkpoint_dir', fallback='checkpoint')

def print_hyper_params():
    print(f'Hyper-params: dropout_rate={dropout_rate}, learning_rate={learning_rate}, batch_size={batch_size}')


# char_set: remove "Il10oO", C->c, K->k, P->p, S->s, W->w, V->v, U->u, X->x, Y->y, Z->z
char_set = "ABDEFGHJMNQRTabcdefghijkmnpqrstuvwxyz23456789"
char_count = 4

image_height = 48
image_width = 128
