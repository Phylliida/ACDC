from collections import defaultdict

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


