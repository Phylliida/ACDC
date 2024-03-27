from collections import defaultdict
from ..acdc import get_pad_token, ACDCDataset

import torch
import random
import numpy as np

# restricts words to only words with the same size tokens
# it choses which size to use based on whichever is most common among the words
# if with_space is true, it considers tokenization when a space is added in front of the word
def restrict_to_most_common_size(tokenizer, words, with_space=False, force_size=None):
    sizes = defaultdict(lambda: 0)
    
    if with_space:
        tokenized_words = [tokenizer.encode(" "  + word) for word in words]
    else:
        tokenized_words = [tokenizer.encode(word) for word in words]
    
    for toks in tokenized_words:
        sizes[len(toks)] += 1
    
    biggest_size, biggest_count = max(sizes.items(), key=lambda x: x[1])
    if force_size:
        biggest_size = force_size
    return [word for toks, word in zip(tokenized_words, words) if len(toks) == biggest_size]


def seed_random(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)  
    np.random.seed(seed)

def repeat_swapped_patch(vec):
    '''
    because every pair (i,i+1) is a patching (uncorrupted, corrupted),
    we want things to be symmetric,
    so we will double up our dataset
    by swapping each pair
    '''
    swapped_vecs = []
    for i in range(0, vec.size()[0], 2):
        swapped_vecs += [vec[i+1:i+2], vec[i:i+1]]
    return torch.cat([vec] + swapped_vecs, dim=0)

def generate_dataset(model,
                  data_generator,
                  num_patching_pairs: int,
                  seed: int,
                  valid_seed: int,
                  constrain_to_answers: bool,
                  has_symmetric_patching: bool=False, 
                  varying_data_lengths=False,
                  insert_bos=True,
                  **kwargs):
    '''
    Given a data_generator and num_examples, generates a ACDCDataset
    data_generator should be a generator that yields tuples of (prompt_str, list_of_correct_answer_strs, list_of_incorrect_answer_strs)
    This calls data_generator twice, once for the dataset and once for the valid dataset
    data_generator should return num_examles total examples.
    These should be paired, so the first (prompt_str, list_of_correct_answer_strs, list_of_incorrect_answer_strs) returned is patched via the second
    (prompt_str, list_of_correct_answer_strs, list_of_incorrect_answer_strs), the third is patched by the fourth, etc.
    Note that before your data_generator is called, the random seeds will be reset according to seed (and valid seed for the valid dataset)
    Args:
        model: Model (this should have a .tokenizer attribute)
        num_examples int: how many datapoints to generate (will also create a valid set with this many datapoints)
        seed int: the random seed used for train set
        valid_seed int: the random seed used for valid set
        constrain_to_answers bool: used when evaluating the model. If true, will only consider relative prs (among correct and incorrect answers). If false, will give prs over all tokens
        has_symmetric_patching bool: if True, will duplicate your dataset swapping every pair. This is useful if your patching is symmetric (i.e., if I patch B->A, it also makes sense to patch A->B)
        varying_data_lengths bool: if True, allows data to have varying length (in terms of number of tokens). Note: if you are doing ACDC on positions this doesn't really make sense
        insert_bos bool: whether to insert bos token before each data point
        **kwargs: extra args to pass into data_generator
    Returns:
        An ACDCDataset containing all your generated data
    '''
    seed_random(seed=seed)
    data, last_token_position, correct, incorrect = data_to_tensors(model=model,
                                                                    data=data_generator(model=model, num_patching_pairs=num_patching_pairs, **kwargs),
                                                                    varying_data_lengths=varying_data_lengths,
                                                                    insert_bos=insert_bos)
    seed_random(seed=valid_seed)
    valid_data, valid_last_token_position, valid_correct, valid_incorrect = data_to_tensors(model=model,
                                                                                            data=data_generator(model=model, num_patching_pairs=num_patching_pairs, **kwargs),
                                                                                            varying_data_lengths=varying_data_lengths,
                                                                                            insert_bos=insert_bos)
    if has_symmetric_patching:
        data = repeat_swapped_patch(data)
        last_token_position = repeat_swapped_patch(last_token_position)
        correct = repeat_swapped_patch(correct)
        incorrect = repeat_swapped_patch(incorrect)
        valid_data = repeat_swapped_patch(valid_data)
        valid_last_token_position = repeat_swapped_patch(valid_last_token_position)
        valid_correct = repeat_swapped_patch(valid_correct)
        valid_incorrect = repeat_swapped_patch(valid_incorrect)
    
    return ACDCDataset(data=data, last_token_position=last_token_position, correct=correct, incorrect=incorrect,
                    valid_data=valid_data, valid_last_token_position=valid_last_token_position, valid_correct=valid_correct, valid_incorrect=valid_incorrect,
                    constrain_to_answers=constrain_to_answers)
    
def add_padding(tokenizer, token_lists):
    '''
    Pads token lists that aren't of the longest length
    with pad_token at the end
    '''
    longest_len = len(max(token_lists, key=lambda x: len(x)))
    padded_answers = []
    pad_token = get_pad_token(tokenizer=tokenizer)
    for tokens in token_lists:
        padded_answers.append(tokens + [pad_token]*(longest_len-len(tokens)))
    return padded_answers

def data_to_tensors(model, data, varying_data_lengths=False, insert_bos=True):
    '''
    Converts data (tuples of (prompt_str, list_of_correct_answer_strs, list_of_incorrect_answer_strs))
    into torch tensors:
    (batched_data, batched_correct, batched_incorrect)
    If some lists of answers are of varying sizes, this will pad with model.tokenizer.pad_token_id
    if some data points are of varying sizes, this will throw an error unless varying_data_lengths=True
    '''
    data_tokens = []
    correct_tokens = []
    incorrect_tokens = []
    bos = [model.tokenizer.bos_token_id]
    if not insert_bos:
        bos = []
    
    for i, (prompt, corrects, incorrects) in enumerate(data):
        if i < 7:
            print(prompt, corrects, incorrects)
        data_tokens.append(bos + model.tokenizer.encode(prompt))
        correct_tokens.append([model.tokenizer.encode(correct)[0] for correct in corrects])
        incorrect_tokens.append([model.tokenizer.encode(incorrect)[0] for incorrect in incorrects])
    
    last_token_position = torch.tensor([len(toks)-1 for toks in data_tokens])
    if varying_data_lengths:
        data_tensor = torch.tensor(add_padding(tokenizer=model.tokenizer, token_lists=data_tokens), device=model.cfg.device)
        for i, (last_token_patch, last_token_corrupted) in enumerate(zip(last_token_position[::2], last_token_position[1::2])):
            if last_token_patch != last_token_corrupted:
                patched = model.to_str_tokens(data_tensor[i*2])
                corrupted = model.to_str_tokens(data_tensor[i*2+1])
                raise ValueError(f'Patch {i*2},{i*2+1} has varying input sizes {last_token_patch+1},{last_token_corrupted+1}\ndata {patched}\ncorrupted {corrupted}\nthese should be the same size')
    else:
        try:
            data_tensor = torch.tensor(data_tokens, device=model.cfg.device)
        except RuntimeError: # thrown if batched_data can't be made a tensor because varying sizes
            typical_len = len(data_tokens[0])
            print(f"first data point is\n{model.to_str_tokens(data_tokens[0])}")
            for toks in data_tokens:
                if not len(toks) == typical_len and not varying_data_lengths:
                    print(f'All data points should be the same number of tokens.\nLength of first data point is {typical_len} however length of this data point is {len(toks)}, this data point is\n{model.to_str_tokens(toks)}\nif this is desired, set varying_data_lengths=True')
            raise ValueError("All data points are not the same length (see above printouts)")
    correct_tensor = torch.tensor(add_padding(tokenizer=model.tokenizer, token_lists=correct_tokens), device=model.cfg.device)
    incorrect_tensor = torch.tensor(add_padding(tokenizer=model.tokenizer, token_lists=incorrect_tokens), device=model.cfg.device)
    return data_tensor, last_token_position, correct_tensor, incorrect_tensor
