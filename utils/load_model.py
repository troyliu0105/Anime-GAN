import os


def load_model_from_params(gen_model, dis_model, save_path):
    all_params = [item for item in os.scandir(save_path) if item.name.lower().endswith('params')]
    # get parameters paths for gen and dis
    gen_params = [item.path for item in all_params if item.name.startswith('generator')]
    dis_params = [item.path for item in all_params if item.name.startswith('discriminator')]
    last_epoch = 0
    if len(gen_params) > 0:
        last_gen_params = sorted(gen_params)[-1]
        gen_model.load_parameters(last_gen_params)
        last_epoch = int(last_gen_params.split('_')[1].split('.')[0])
    if len(dis_params) > 0:
        last_dis_params = sorted(dis_params)[-1]
        dis_model.load_parameters(last_dis_params)
        _ep = int(last_dis_params.split('_')[1].split('.')[0])
        if _ep < last_epoch:
            last_epoch = _ep
    return last_epoch
