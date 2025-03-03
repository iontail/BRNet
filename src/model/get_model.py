from src.model.basic_modules import Generator, Discriminator

def get_model(args):
    if args.data_name == '256':
        img_res = 256
    elif args.data_name == '512':
        img_res = 512
    G = Generator(z_dim=img_res, c_dim=0, w_dim=img_res, img_resolution=img_res, img_channels=3, use_edge = True, use_line = False, use_ln = True, few_high = False)
    D = Discriminator(c_dim=0, img_resolution=img_res, img_channels=3)

    print(f'Generator: {count_parameters(G)}')
    print(f'Discriminator: {count_parameters(D)}')
    return G, D

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)