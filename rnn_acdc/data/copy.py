# todo: symmetric? probably just substitute a different random token
def copy_generator(tokenizer, num_examples, copy_seq_len, seed=27):
    random.seed(27)
    bos_token = tokenizer.bos_token_id
    torch.random.manual_seed(seed)
    first_len = None
    for i in range(num_examples):
        # generate one twice as big, it'll be messed up when decoding so that way we can clip it at desired token len
        while True:
            data = torch.randint(low=1000, high=tokenizer.vocab_size-1000, size=(copy_seq_len*2,))
            prompt = tokenizer.decode(data).encode("ascii", "ignore").decode("ascii", "ignore")
            tokens = tokenizer.encode(prompt)[:copy_seq_len]
            answer = tokenizer.decode([tokens[-1]])
            full_tokens = tokens + tokens[:-1]
            full_prompt = tokenizer.decode(full_tokens)
            full_tokens = tokenizer.encode(full_prompt)
            full_len = len(full_tokens)
            if first_len is None:
                first_len = full_len
            # retry until we match the size
            if not full_len == first_len:
                continue
            else:
                yield full_prompt, [answer], [tokenizer.pad_token]
                break