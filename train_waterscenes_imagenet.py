from train import get_args, train


if __name__ == '__main__':
    opt = get_args()
    opt.project = 'waterscenes_imagenet'
    opt.batch_size = 16
    opt.num_epochs = 100
    opt.num_gpus = 1
    opt.gpu_ids = '1'
    opt.plots = False
    train(opt)
