import argparse
import time
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.onnx
from torch.utils.data import DataLoader

import data
import model
import _pickle as cPickle
from data import Batch
from transformer_model import make_model


# Creates sequences of (seq_len+1) words from all the sentences of the raw
# data set. Input is seq_len words, and the ground truth output is the
# (seq_len+1)th word.

def create_data_set(data, seq_len):
    x_set = []
    y_set = []

    for sent in data:
        for i, word in enumerate(sent):
            # exclude sentences which have less than (seq_len+1) words
            if i + seq_len + 1 >= len(sent):
                break
            x_sent = sent[i:i + seq_len]
            y_sent = [sent[i + seq_len]]
            x_set.append(x_sent)
            y_set.append(y_sent)

    return x_set, y_set


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def get_batch_alt(source, batch):
    data = source[batch:batch + args.bptt - 1]
    target = source[batch + args.bptt - 1].view(-1)
    return data, target


def evaluate(model, dataloader, criterion):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    # if args.model != 'Transformer':
    #     hidden = model.init_hidden(eval_batch_size)
    count = 0
    with torch.no_grad():
        for iter, data in enumerate(dataloader):
            inputs = data[:, 0:args.bptt]
            targets = data[:, args.bptt]

            inputs, targets = inputs.to(device), targets.to(device)
            # if args.model == 'Transformer':
            #     output = model(data)
            #     output = output.view(-1, ntokens)
            # else:
            #     output, hidden = model(data, hidden)
            #     hidden = repackage_hidden(hidden)
            output = model(inputs)
            total_loss += criterion(output, targets).item()
            count = count + 1
    return total_loss / count


def train(model, dataloader, criterion, optimizer):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    # if args.model != 'Transformer':
    #     hidden = model.init_hidden(args.batch_size)

    for iter, data in enumerate(dataloader):
        inputs = data[:, 0:args.bptt]
        targets = data[:, args.bptt]

        inputs, targets = inputs.to(device), targets.to(device)


        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            targets = targets.unsqueeze(1)
            batch_trans = Batch(inputs, targets, 0)
            output = model(batch_trans.src, batch_trans.trg,
                           batch_trans.src_mask, batch_trans.trg_mask)
            output = model.generator(output)
            targets = batch_trans.trg_y
            targets = targets.squeeze(1)
            output = output.squeeze(1)
        else:
        #     hidden = repackage_hidden(hidden)
        #     output, hidden = model(data, hidden)
            output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()
        if iter % args.log_interval == 0 and iter > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:4f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, iter, (len(dataloader.dataset) // args.batch_size), lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    # hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, dummy_input, path)


# Loop over epochs.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 LNN/RNN/LSTM/GRU/Transformer Language Model')
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer, FNN)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=6,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')
    parser.add_argument('--nhead', type=int, default=2,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--dry-run', action='store_true',
                        help='verify the code and the model')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for DataLoader')
    parser.add_argument('--patience', type=int, default=5,
                        help='number of workers for DataLoader')
    parser.add_argument('--min_count', type=int, default=0,
                        help='to remove words that occur less than min_count number of times')
    parser.add_argument('--corpus_pickle', action='store_true',
                        help='load corpus object from existing pickle file or save corpus to new pickle file')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda.")

    device = torch.device("cuda" if args.cuda else "cpu")

    # load corpus from pickle file if it exists. removing rare tokens from data set
    # is time consuming. Set --min_count=0 if not using pickle file.
    if os.path.exists('./corpus.pkl') and args.corpus_pickle:
        print('Loading corpus from pickle file...')
        with open(r'corpus.pkl', 'rb') as myfile:
            corpus = cPickle.load(myfile)
    else:
        corpus = data.Corpus(args.data, args.min_count)
        if args.corpus_pickle:
            print("storing corpus object in corpus.pkl...")
            with open(r'corpus.pkl', 'wb') as myfile:
                cPickle.dump(corpus, myfile)

    eval_batch_size = 10
    # train_data = batchify(corpus.train, args.batch_size)
    # val_data = batchify(corpus.valid, eval_batch_size)
    # test_data = batchify(corpus.test, eval_batch_size)

    ###############################################################################
    # Load data
    ###############################################################################

    x_train, y_train = create_data_set(corpus.train_sent, args.bptt)
    x_valid, y_valid = create_data_set(corpus.valid_sent, args.bptt)
    x_test, y_test = create_data_set(corpus.test_sent, args.bptt)

    x_train = np.array(x_train).astype(np.int64)
    y_train = np.array(y_train).astype(np.int64)
    x_valid = np.array(x_valid).astype(np.int64)
    y_valid = np.array(y_valid).astype(np.int64)
    x_test = np.array(x_test).astype(np.int64)
    y_test = np.array(y_test).astype(np.int64)

    train_set = np.concatenate((x_train, y_train), axis=1)
    valid_set = np.concatenate((x_valid, y_valid), axis=1)
    test_set = np.concatenate((x_test, y_test), axis=1)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)

    ###############################################################################
    # Build the model
    ###############################################################################

    ntokens = len(corpus.dictionary)

    if args.model == 'Transformer':
        #model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout) \
        #    .to(device)
        model = make_model(ntokens, ntokens, N=6, d_model=args.emsize, h=args.nhead,
                                             dropout=args.dropout) \
                .to(device)
    elif args.model == 'FNN':
        model = model.FNNModel(ntokens, args.emsize, args.bptt, args.nhid, args.dropout, args.tied).to(device)
    else:
        model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied) \
            .to(device)

    lr = args.lr
    best_val_loss = None

    criterion = nn.NLLLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    patience_counter = 0
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train(model, train_loader, criterion, optimizer)
            val_loss = evaluate(model, valid_loader, criterion)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'
                  .format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                patience_counter = 0
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                patience_counter += 1
                print('Incrementing early stopping counter. Current value is ', patience_counter)
                print('-' * 89)

            if patience_counter >= args.patience:
                print("Stopping early since validation loss didn't improve for {} epochs".format(args.patience))
                print('-' * 89)
                break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(model, test_loader, criterion)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('| best valid loss {:5.2f} | best valid ppl {:8.2f}'.format(
        best_val_loss, math.exp(best_val_loss)))
    print('=' * 89)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)



