from torch.optim import *

def get_optimizer(model, args):
    if args.optimizer == 'Adam':
        return Adam(model.module.transformer.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        return AdamW(model.module.transformer.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    else:
        raise NotImplementedError