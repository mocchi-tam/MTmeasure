# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.dataset import convert

import measure_performance

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--train', '-t', type=int, default=1,
                        help='If negative, skip training')
    parser.add_argument('--resume', '-r', type=int, default=1,
                        help='If positive, resume the training from snapshot')
    args = parser.parse_args()
    
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    
    flag_train = False if args.train < 0 else True
    flag_resum = False if args.resume < 0 else True
    n_epoch = args.epoch if flag_train == True else 1
    
    tsm = MTModel(args.gpu, flag_train, flag_resum, n_epoch, args.batchsize)
    tsm.run()

class MTNNet(chainer.Chain):
    def __init__(self, n_mid, n_out):
        super(MTNNet, self).__init__()
        with self.init_scope():
            self.lin1 = L.Linear(None, n_mid)
            self.lin2 = L.Linear(None, n_out)
        
    def __call__(self, x):
        h1 = self.lin1(x)
        h2 = F.relu(h1)
        h3 = F.dropout(h2)
        
        y = self.lin2(h3)
        return y
    
    def loss(self, x, t):
        y = self(x)
        loss = F.softmax_cross_entropy(y,t)
        self.accuracy = F.accuracy(y,t)
        return y, loss

class MTModel():
    def __init__(self, gpu, flag_train, flag_resum, n_epoch, batchsize):
        self.n_epoch = n_epoch
        self.flag_train = flag_train
        self.flag_resum = flag_resum
        self.gpu = gpu
        
        self.model = MTNNet(256, 10)
        
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            self.model.to_gpu()
        
        if self.flag_train:
            self.optimizer = chainer.optimizers.Adam()
            self.optimizer.setup(self.model)
        
        if self.flag_resum:
            try: 
                chainer.serializers.load_npz('./net/net.model', self.model)
                chainer.serializers.load_npz('./net/net.state', self.optimizer)
                print('successfully resume model')
            except:
                print('ERROR: cannot resume model')
        
        # prepare dataset
        train, test = chainer.datasets.get_mnist()
        
        self.N_train = len(train)
        self.N_test = len(test)
        
        self.train_iter = chainer.iterators.SerialIterator(train, batchsize,
                                                           repeat=True, shuffle=True)
        self.test_iter = chainer.iterators.SerialIterator(test, self.N_test,
                                                          repeat=False, shuffle=False)
        
    def run(self):
        sum_accuracy = 0
        sum_loss = 0
        
        mtp = measure_performance.MTPerform(self.gpu)
        mtp.start()
        
        while self.train_iter.epoch < self.n_epoch:
            # train phase
            batch = self.train_iter.next()
            if self.flag_train:
                # step by step update
                x, t = convert.concat_examples(batch, self.gpu)
                
                self.model.cleargrads()
                y, loss = self.model.loss(x, t)
                loss.backward()
                self.optimizer.update()
                
                sum_loss += float(loss.data) * len(t.data)
                sum_accuracy += float(self.model.accuracy.data) * len(t.data)
            
            # test phase
            if self.train_iter.is_new_epoch:
                print('epoch: ', self.train_iter.epoch)
                print('train mean loss: {}, accuracy: {}'.format(
                        sum_loss / self.N_train, sum_accuracy / self.N_train))
                
                sum_accuracy = 0
                sum_loss = 0
                mtp.reset()
                
                for batch in self.test_iter:
                    x, t = convert.concat_examples(batch, self.gpu)
                    
                    with chainer.using_config('train', False), chainer.no_backprop_mode():
                        y, loss = self.model.loss(x, t)
                        
                    sum_loss += float(loss.data) * len(t.data)
                    sum_accuracy += float(self.model.accuracy.data) * len(t.data)
                
                mtp.write('data/performance' + str(self.train_iter.epoch) + '.csv')
                self.test_iter.reset()
                print('test mean  loss: {}, accuracy: {}'.format(
                        sum_loss / self.N_test, sum_accuracy / self.N_test))
                
                sum_accuracy = 0
                sum_loss = 0
        
        mtp.stop()
        
        try:
            chainer.serializers.save_npz('net/net.model', self.model)
            chainer.serializers.save_npz('net/net.state', self.optimizer)
            print('Successfully saved model')
        except:
            print('ERROR: saving model ignored')

if __name__ == '__main__':
    main()