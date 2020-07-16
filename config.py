import configparser
cfg_parser = configparser.ConfigParser()
cfg_parser.read('config.ini')


# Hyper-params:
dropout_rate = cfg_parser.getfloat('hyper-param', 'dropout_rate', fallback=0.25)
learning_rate = cfg_parser.getfloat('hyper-param', 'learning_rate', fallback=0.0001)
batch_size = cfg_parser.getint('hyper-param', 'batch_size', fallback=32)

# Environment
checkpoint_dir = cfg_parser.get('env', 'checkpoint_dir', fallback='checkpoint')


# char_set: remove "Il10oO", C->c, K->k, P->p, S->s, W->w, V->v, U->u, X->x, Y->y, Z->z
char_set = "ABDEFGHJMNQRTabcdefghijkmnpqrstuvwxyz23456789"
char_count = 4

image_height = 48
image_width = 128


def print_hyper_params():
    print(f'Hyper-params: dropout_rate={dropout_rate}, learning_rate={learning_rate}, batch_size={batch_size}')


if __name__ == '__main__':
    print_hyper_params()
    print(f'checkpoint_dir={checkpoint_dir}')
