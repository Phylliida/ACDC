import torch

def decode_and_encode(tokenizer, tokens):
    '''
    Gets rid of weird encoding issues by encoding and decoding
    The tokens will be different that's okay and intentional
    '''
    prompt = tokenizer.decode(tokens).encode("ascii", "ignore").decode("ascii", "ignore")
    return tokenizer.encode(prompt)

def copy_data_generator(tokenizer, num_patching_pairs, copy_seq_len):
    first_len = None
    for i in range(num_patching_pairs):
        while True:
            # generate one twice as big, it'll be messed up when decoding but that's ok just clip it at desired token len
            data = torch.randint(low=1000, high=tokenizer.vocab_size-1000, size=(copy_seq_len*2,))
            tokens = decode_and_encode(data)[:copy_seq_len]

            corrupted_token = torch.randint(low=1000, high=tokenizer.vocab_size-1000, size=(1,))[0]
            tokens_corrupted = tokens.clone()
            tokens_corrupted[-1] = corrupted_token
            
            full_tokens = decode_and_encode(tokens + tokens[:-1])
            full_corrupted_tokens = decode_and_encode(tokens_corrupted + tokens_corrupted[:-1])

            # make sure all the lengths match
            if len(full_tokens) != len(full_corrupted_tokens):
                continue
            if first_len is None:
                first_len = len(full_tokens)
            elif len(full_tokens) != first_len:
                continue
            
            prompt = tokenizer.decode(full_tokens)
            corrupted_prompt = tokenizer.decode(full_corrupted_tokens)
            answer = tokenizer.decode([tokens[-1]])
            corrupted_answer = tokenizer.decode([tokens_corrupted[-1]])
            yield prompt, [answer], [corrupted_answer]
            yield corrupted_prompt, [corrupted_answer], [answer]
