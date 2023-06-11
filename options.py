import argparse


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--dataroot', type=str, default=r'./datasets/')
        self.parser.add_argument('--save_dir', type=str, default=r'./results/')      
        self.parser.add_argument('--epoch_num', type=int, default=40)
        self.parser.add_argument('--loss', type=str, default='mse')  #mse,l1,sm_l1
        self.parser.add_argument('--mertics', type=str, default='mse')
        # data loader related
        self.parser.add_argument('--is_norm', type=bool, default=False)
        self.parser.add_argument('--is_dna', type=bool, default=False)
        self.parser.add_argument('--augment', type=bool, default=True)
        self.parser.add_argument('--pb_aug', type=str, default='pbda')
        self.parser.add_argument('--load_weights', type=bool, default=True)
        self.parser.add_argument('--weights_path', type=str,
                                 default=r'')
        self.parser.add_argument('--batch_size', type=int, default=2)
        self.parser.add_argument('--val_intel', type=int, default=1)
        self.parser.add_argument('--adv_loss_weight', type=float, default=0.1)
        self.parser.add_argument('--learn_rate_u', type=float, default=0.0001)
        self.parser.add_argument('--learn_rate_g', type=float, default=0.001)
        self.parser.add_argument('--learn_rate_d', type=float, default=0.001)        
        self.parser.add_argument('--focal_loss', type=str, default='false')
        self.parser.add_argument('--note', type=str, default='in_fch=32')
        self.parser.add_argument('--dropout', type=str, default='true')
        self.parser.add_argument('--warm_up', type=bool, default=False)
        self.parser.add_argument('--inch', type=int, default=5)

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        # for name, value in sorted(args.items()):
        #     print('%s: %s' % (str(name), str(value)))
        return self.opt
