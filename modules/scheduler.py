from torch.optim.lr_scheduler import *

def get_scheduler(args, optimizer):
    if args.lr_scheduler == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr/100)
    elif args.lr_scheduler == 'ExponentialLR':
        return ExponentialLR(optimizer, gamma=0.9)
    elif args.lr_scheduler == 'StepLR':
        return StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.lr_scheduler == 'MultiStepLR':
        return MultiStepLR(optimizer, milestones=[args.num_epochs//2, args.num_epochs*3//4, args.num_epochs*7//8], gamma=0.5)
    elif args.lr_scheduler == 'LambdaLR':
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)
    else:
        return None