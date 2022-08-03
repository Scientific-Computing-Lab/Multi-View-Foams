import yaml

with open(r'/home/nadavsc/Desktop/projects/targets/config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    data_dir = config['data_dir']
    images_dir = config['images_dir']
    full_groups_dir = config['full_groups_dir']
    preprocess_dir = config['preprocess_dir']
    augmentation_dir = config['augmentation_dir']
    train_dir = config['train_dir']
    test_dir = config['test_dir']
    models2_dir = config['models2_dir']
