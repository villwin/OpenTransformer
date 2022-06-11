try:
    from frob import FactorizedConv, frobdecay
except ImportError:
    print("Failed to import factorization")
try:
    from frob import FactorizedLinear, batch_spectral_init, frobenius_norm, patch_module, non_orthogonality
except ImportError:
    print("Failed to import factorization")
from torch import nn
def patch_transformer(args, model):
    convs = ['conv1', 'conv2']
    modules, namelists = [], []
    for module3 in list(model.frontend.modules()):
        namelist = []
        for name in convs:
            if hasattr(module3, name):
                namelist.append(name)
        if namelist:
            modules.append(module3)
            namelists.append(namelist)
        if hasattr(module3, 'output_layer'):
            patch_module(module3, 'output_layer', FactorizedLinear,
                     rank_scale=args.rank_scale,
                     init='spectral' if args.spectral else lambda X: nn.init.uniform_(X, -0.1, 0.1))
    for module, namelist in zip(modules, namelists):
        for name in namelist:
            patch_module(module, name, FactorizedConv,
                         rank_scale=args.rank_scale, square=args.square,
                         init='spectral' if args.spectral else lambda X: nn.init.kaiming_normal_(X),
                         square_init=lambda I: I + nn.init.normal_(I.clone(), 0.0,
                                                                   args.residual) if args.residual else I)
    for block in model.encoder.blocks:
        for name in ['w_1', 'w_2']:
            patch_module(block.pre_ffn, name, FactorizedLinear,
                            rank_scale=args.rank_scale,
                            init='spectral' if args.spectral else lambda X: nn.init.uniform_(X, -0.1, 0.1))
        for name in ['w_1', 'w_2']:
                    patch_module(block.post_ffn, name, FactorizedLinear,
                                 rank_scale=args.rank_scale,
                                 init='spectral' if args.spectral else lambda X: nn.init.uniform_(X, -0.1, 0.1))
        for name in ['x_proj']:
                    patch_module(block.mha, name, FactorizedLinear,
                                 rank_scale=args.rank_scale,
                                 init='spectral' if args.spectral else lambda X: nn.init.uniform_(X, -0.1, 0.1))
        for name in ['dense']:
                    patch_module(block.mha, name, FactorizedLinear,
                                 rank_scale=0.5,
                                 init='spectral' if args.spectral else lambda X: nn.init.uniform_(X, -0.1, 0.1))
        for name in ['pointwise_conv1','pointwise_conv2']:
                    patch_module(block.conv, name, FactorizedLinear,
                                 rank_scale=args.rank_scale,
                                 init='spectral' if args.spectral else lambda X: nn.init.uniform_(X, -0.1, 0.1))
    '''patch_module(model.decoder, 'output_layer', FactorizedLinear,
                 rank_scale=args.rank_scale,
                 init='spectral' if args.spectral else lambda X: nn.init.uniform_(X, -0.1, 0.1))'''
    for block in model.decoder.blocks:
        for name in ['w_1', 'w_2']:
                    patch_module(block.feed_forward, name, FactorizedLinear,
                                 rank_scale=args.rank_scale,
                                 init='spectral' if args.spectral else lambda X: nn.init.uniform_(X, -0.1, 0.1))
        for name in ['output_proj']:
            patch_module(block.slf_attn, name, FactorizedLinear,
                         rank_scale=0.5,
                         init='spectral' if args.spectral else lambda X: nn.init.uniform_(X, -0.1, 0.1))
        for name in ['qvk_proj']:
            patch_module(block.slf_attn, name, FactorizedLinear,
                         rank_scale=0.5,
                         init='spectral' if args.spectral else lambda X: nn.init.uniform_(X, -0.1, 0.1))
        '''for name in ['dense']:
            patch_module(block.slf_attn, name, FactorizedLinear,
                         rank_scale=0.5,
                         init='spectral' if args.spectral else lambda X: nn.init.uniform_(X, -0.1, 0.1))'''
        for name in ['q_proj', 'vk_proj','output_proj']:
            patch_module(block.src_attn, name, FactorizedLinear,
                         rank_scale=0.5,
                         init='spectral' if args.spectral else lambda X: nn.init.uniform_(X, -0.1, 0.1))

        '''for name in ['vk_proj', 'q_proj']:
                        patch_module(block.src_attn, name, FactorizedLinear,
                                     rank_scale=args.rank_scale,
                                     init='spectral' if args.spectral else lambda X: nn.init.uniform_(X, -0.1, 0.1))'''
