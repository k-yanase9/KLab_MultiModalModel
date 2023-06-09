from torch.optim.lr_scheduler import *

def get_scheduler(args, optimizer):
    if args.lr_scheduler == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)
    elif args.lr_scheduler == 'linear':
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - (epoch / args.num_epochs))
    elif args.lr_scheduler == 'exponential':
        return ExponentialLR(optimizer, gamma=0.9)
    elif args.lr_scheduler == 'step':
        return StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        return None