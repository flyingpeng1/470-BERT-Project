# This is the warp loss implementation borrowed with small modifications from 
# https://github.com/NegatioN/WARP-Pytorch/

# For some reason the module was not working, but it is a relatively simple file to download and use manually.

import torch
import time


def num_tries_gt_zero(scores, batch_size, max_trials, max_num, device, cuda_bool, time_map):
    '''
    scores: [batch_size x N] float scores
    returns: [batch_size x 1] the lowest indice per row where scores were first greater than 0. plus 1
    '''
    #start_time = time.time()

    tmp = scores.gt(0)

    #time_map["warp_p1"]+=time.time()-start_time
    #start_time = time.time()

    tmp = tmp.nonzero()

    #time_map["warp_p2"]+=time.time()-start_time
    #start_time = time.time()

    tmp = tmp.t()

    #time_map["warp_p3"]+=time.time()-start_time
    #start_time = time.time()


    # We offset these values by 1 to look for unset values (zeros) later
    values = tmp[1] + 1
    # TODO just allocate normal zero-tensor and fill it?
    # Sparse tensors can't be moved with .to() or .cuda() if you want to send in cuda variables first

    #time_map["warp_p4"]+=time.time()-start_time

    if cuda_bool :
        t = torch.cuda.sparse.LongTensor(tmp, values, torch.Size((batch_size, max_trials+1))).to_dense()
    else:
        t = torch.sparse.LongTensor(tmp, values, torch.Size((batch_size, max_trials+1))).to_dense()



    t[(t == 0)] += max_num # set all unused indices to be max possible number so its not picked by min() call

    tries = torch.min(t, dim=1)[0]

    return tries

def warp_loss(positive_predictions, negative_predictions, num_labels, device, time_map, cuda_bool):
    '''
    positive_predictions: [batch_size x 1] floats between -1 to 1
    negative_predictions: [batch_size x N] floats between -1 to 1
    num_labels: int total number of labels in dataset (not just the subset you're using for the batch)
    device: pytorch.device
    '''

    batch_size, max_trials = negative_predictions.size(0), negative_predictions.size(1)

    offsets, ones, max_num = (torch.arange(0, batch_size, 1, device=device).long() * (max_trials + 1),
                              torch.ones(batch_size, 1, device=device).float(),
                              batch_size * (max_trials + 1) )



    sample_scores = (1 + negative_predictions - positive_predictions)
    # Add column of ones so we know when we used all our attempts, This is used for indexing and computing should_count_loss if no real value is above 0
    sample_scores, negative_predictions = (torch.cat([sample_scores, ones], dim=1),
                                           torch.cat([negative_predictions, ones], dim=1))


    tries = num_tries_gt_zero(sample_scores, batch_size, max_trials, max_num, device, cuda_bool, time_map)


    attempts, trial_offset = tries.float(), (tries - 1) + offsets
    loss_weights, should_count_loss = ( torch.log(torch.floor((num_labels - 1) / attempts)),
                                        (attempts <= max_trials).float()) #Don't count loss if we used max number of attempts

    losses = loss_weights * ((1 - positive_predictions.view(-1)) + negative_predictions.view(-1)[trial_offset]) * should_count_loss


    return losses.sum()