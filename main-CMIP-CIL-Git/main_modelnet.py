import json
import argparse
from trainer import train


def main():
    '''
    # all list
    model_list = ['finetune', 'ewc', 'lwf', 'replay', 'gem', 'icarl', 'bic', 'wa', 'podnet', 'der', 'pass', 'rmm-icarl',
                  'il2a', 'ssre', 'fetril', 'coil', 'foster', 'memo', 'beefiso', 'simplecil']
    '''

    # icarl wa podnet beefiso simplecil foster already yanzheng
    model_list = ['oursmethod']
    seed_list = [[1993]]
    for old_data in [True]:
        for model in model_list:
            for seed in seed_list:
                if old_data == False and model =='replay':
                    continue
                print('-------A BaseLine method is Running-------')
                print('-------DataSet: modelnet40-------')
                print('-------Model: ' + str(model) + '-------')
                print('-------Seed: ' + str(seed) + '-------')
                print('-------'+ str(old_data) +' Using Old Data-------')
                args = setup_parser().parse_args()
                args.config = './exps/' + str(model) + '.json'
                param = load_json(args.config)
                args = vars(args)  # Converting argparse Namespace to a dict.
                args.update(param)  # Add parameters from json
                args['dataset'] = 'm_modelnet40'
                if 'fixed_memory' in args:
                    args['fixed_memory'] = True
                if old_data == True:
                    if 'memory_size' in args:
                        args['memory_size'] = 8001
                    if 'memory_per_class' in args:
                        args['memory_per_class'] = 20
                else:
                    if 'memory_size' in args:
                        args['memory_size'] = 0
                    if 'memory_per_class' in args:
                        args['memory_per_class'] = 0
                args['init_cls'] = 4
                args['increment'] = 4
                args['model_name'] = str(model)
                args['convnet_type'] = 'cross_model'
                args['device'] = ['0']
                args['seed'] = seed
                args['k'] = 20
                args['emb_dims'] = 1024
                args['pretrain'] = False
                args['pretrain_modelpath'] = "/root/autodl-tmp/PyCIL_CrossModal/pretrain/m_modelnet40_1993/best_model_epo99_tem002_lr0001.pth"
                print(args)
                train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/icarl.json',
                        help='Json file of settings.')

    return parser


if __name__ == '__main__':
    main()
