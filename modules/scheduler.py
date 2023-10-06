from torch.optim.lr_scheduler import *
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


def get_scheduler(args, optimizer):
    warmup_steps = int(args.num_steps * args.warmup_rate)
    if args.lr_scheduler == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr / 100)
    elif args.lr_scheduler == 'ExponentialLR':
        return ExponentialLR(optimizer, gamma=0.9)
    elif args.lr_scheduler == 'StepLR':
        return StepLR(optimizer, step_size=args.num_epochs // 5 if args.num_epochs // 5 != 0 else 1, gamma=0.5)
    elif args.lr_scheduler == 'MultiStepLR':
        return MultiStepLR(
            optimizer,
            milestones=[
                args.num_epochs // 2 if args.num_epochs // 2 != 0 else 1,
                args.num_epochs * 3 // 4 if args.num_epochs * 3 // 4 != 0 else 1,
                args.num_epochs * 7 // 8 if args.num_epochs * 7 // 8 != 0 else 1,
            ],
            gamma=0.5,
        )
    elif args.lr_scheduler == 'LambdaLR':
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99**epoch)
    elif args.lr_scheduler == 'LinearWarmup':
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.num_steps)
    elif args.lr_scheduler == 'CosineWarmup':
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.num_steps)
    else:
        return None
