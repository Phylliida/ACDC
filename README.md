# ACDC
ACDC (Automated Circuit Discovery) alternative implementation.


## Install

```
pip install git+https://github.com/Phylliida/ACDC.git
```

## Usage

I recommend looking at the notebook in [Mamba Layers](https://github.com/Phylliida/ACDC/blob/main/examples/Mamba%20Layers.ipynb) for a minimal example of usage.

In general, this library has two pieces, the *dataset* and the *acdc edges*

## Dataset

Here's the specification for an ACDCDataset

```python
ACDCDataset:
    data: Float[torch.Tensor, "batch context_len"] # input tokens
    last_token_position: Float[torch.Tensor, "batch"] # position of last token in each data sample (used if they have varying lengths and some are padded)
    correct: Float[torch.Tensor, "batch num_correct_answers"] # each data point has a list of correct answer tokens
    incorrect: Float[torch.Tensor, "batch num_incorrect_answers"] # each data point has a list of incorrect answer tokens
    # same but for validation data
    valid_data: Float[torch.Tensor, "batch context_len"]
    valid_last_token_position: Float[torch.Tensor, "batch"]
    valid_correct: Float[torch.Tensor, "batch num_correct_answers"]
    valid_incorrect: Float[torch.Tensor, "batch num_incorrect_answers"]
    # if True, prs will be computed as a softmax over all possible answer (incorrect or correct) tokens's logits.
    # if False, prs will be computed as a softmax over all possible tokens's logits in the vocab
    constrain_to_answers: bool
```

data is expected to be provided in pairs ((0,1), (2,3), (4, 5), etc.) where each pair corresponds to

(uncorrupted, corrupted)

so for example, data[0] might be

"Lately, John and Susan went to the store. John gave an apple to" (tokenized)

and data[1] would be

"Lately, John and Susan went to the store. Susan gave an apple to" (tokenized)

This means that we will patch in data[1]'s activations into data[0].

If you don't want to bother with tokenization and padding, there's a utility function

```python
from acdc.data.utils import generate_dataset
```

Here's it's doc:

```python
def generate_dataset(model, # your model, should have a .tokenizer property
    data_generator, # your data_generator function
    num_patching_pairs: int, # how many pairs to generate
    seed: int, # random seed for data
    valid_seed: int, # random seed for valid data
    constrain_to_answers: bool, # ACDCDataset's constrain_to_answers
    has_symmetric_patching: bool=False,
    varying_data_lengths=False, # if true, data can vary in length (number of tokens)
    insert_bos=True, # if True, will add BOS token as the first token
    **kwargs):
```

this expects a data_generator that looks something like this

```python
def simple_data_generator(tokenizer, num_patching_pairs):
    for pair in range(num_patching_pairs):
        # generate data
        # ...
        yield uncorrupted_prompt, uncorrupted_correct, uncorrupted_incorrect
        yield corrupted_prompt, corrupted_correct, corrupted_incorrect
```

where
- `uncorrupted_prompt` and `corrupted_prompt` are str, these are your prompts
- `uncorrupted_correct` and `corrupted_correct` are lists of str, these are the correct answers
- `uncorrupted_incorrect` and `corrupted_incorrect` are lists of str, these are the incorrect answers

Note that this expects it in pairs,

`uncorrupted` is the data we are going to patch on, and `corrupted` is the data we are going to patch into uncorrupted (more on this in the edges section)

You can see the [data directory](https://github.com/Phylliida/ACDC/tree/main/acdc/data) for some examples of generators for common tasks.

This function will automatically convert your data into torch tensors and do padding as needed.

### Notes

#### has_symmetric_patching

There are many types of patching that are symmetric, i.e., if you have a patching pair (A,B) it also makes sense to patch (B,A).

If you set `has_symmetric_patching=True`, your dataset will be made twice as long, where every (A,B) in the first half will be (B,A) in the second half.

#### varying_data_lengths

The reason this is not default True is because there are sorts of edges that depend on token position. If you use these, having varying data lengths doesn't make
sense, so it's useful to be warned about this and have to enable it manually.

# Edges

To run an ACDC run, you need to specify all of the edges that might be patched. Edges look like this
```python
from acdc import Edge

i = 3
j = 5
# edge from layer i to layer j
edge = Edge(
    label=f'{i}->{j}'
    input_node=f"layer {i}",
    input_hook=(f'blocks.{i}.hook_out_proj', storage_hook),
    output_node=f"layer {j}",
    output_hook=(f'blocks.{j}.hook_layer_input', resid_patching_hook),
)
```

`label` is optional, and is just used for drawing the graph.

`input_node` and `output_node` are strings, and are the nodes this edge goes between.
These can be arbitrary string labels, and this library will hook up any edges with matching node names.

`input_hook` and `output_hook` are both tuples of `(hook_name, hook_func)` like you might see in [TransformerLens](https://github.com/neelnanda-io/TransformerLens).
The `hook_name` is the name of a `TransformerLens` hook, while the `hook_func` is the function that'll be called.
These hooks will be called whenever that edge is considered "patched", so you should write code accordingly.

For example, here are the hooks for the above example, that correspond to edge-based ACDC:

```python
global storage
storage = {}
def storage_hook(
    x,
    hook: HookPoint,
    output_hook_name: str,
    batch_start: int,
    batch_end: int,
):
    global storage
    storage[hook.name] = x
    return x

def resid_patching_hook(
    x,
    hook: HookPoint,
    input_hook_name: str,
    batch_start: int,
    batch_end: int,
):
    global storage
    x_uncorrupted = storage[input_hook_name][batch_start:batch_end:2]
    x_corrupted = storage[input_hook_name][batch_start+1:batch_end:2]
    x[batch_start:batch_end:2] = x[batch_start:batch_end:2] - x_uncorrupted + x_corrupted
    return x
```


You can see some extra parameters provided to these hooks:

`output_hook_name` and `input_hook_name` are the corresponding names of the other side of the edge, which are frequently needed so I provide them.

`batch_start` and `batch_end` are important because this library runs things in a batched manner, duplicating your dataset `batch_size` number of times and running a single forward pass. If you don't want to bother with these, you can see `batched=False` in your `ACDCConfig` (see below) and then `batch_start` and `batch_end` will not be provided.

Looking at what they are doing:

`storage_hook` is just being used to store some activation values that are later used for `resid_patching_hook`.

In `resid_patching_hook`, this line is the important bit:

```python
x_uncorrupted = storage[input_hook_name][batch_start:batch_end:2]
x_corrupted = storage[input_hook_name][batch_start+1:batch_end:2]
```    

Because our data comes in pairs (patching, corrupted), this retreives every even data point for uncorrupted (0,2,4,...) and every odd data point for corrupted (1,3,5,...)

Now we can do

```python
x[batch_start:batch_end:2] = x[batch_start:batch_end:2] - x_uncorrupted + x_corrupted
```

remember that `batch_start:batch_end:2` means that we are only modifying the first pair, you should leave corrupted alone.

By subtracting `x_uncorrupted`, we remove the output of that layer.

By adding `x_corrupted`, this now acts like that layer if it was corrupted.

You can see [Mamba Layers](https://github.com/Phylliida/ACDC/blob/main/examples/Mamba%20Layers.ipynb) for a full example. It includes edges:
- between every layer
- for embed -> each layer
- each layer -> output node 

### Notes:

`input_hook` and `output_hook` are optional. If both are not provided (or set to None), the edge will never be patched.
(this is useful if you just need to specify two nodes are connected by something you never patch)

Sometimes you need more than one hook, `input_hook` and `output_hook` can optionally be lists of (hook_name, hook_func) tuples.
In this case, the `input_hook_name` or `output_hook_name` parameters would be lists of strings respectively.

# ACDC Config

Now that you have an `ACDCDataset` and a list of `Edge`, you can create an ACDCConfig:

```python
cfg = ACDCConfig(
    thresh = 0.00001, # thresh in ACDC, if you try to patch an edge and your metric is affected less than this value, it will be patched
    rollback_thresh = 0.00001, # used in rollback (below)
    metric=your_metric,
    # extra inference args to pass to model
    model_kwargs={},
    # these are required strings that are your source node and sink node
    # any edges that no longer have a path from input_node to output_node will be pruned 
    input_node='input',
    output_node='output',
    # batch size for evaluating data points
    batch_size=3,
    # how much info to print
    log_level=LOG_LEVEL_INFO,
    # if False, will be equivalent to batch_size=1
    batched = True,
    # set these two to false to use traditional ACDC
    # recursive will try patching multiple at a time (this is faster sometimes)
    recursive = True,
    # try_patching_multiple_at_same_time will evaluate many different patchings before commiting to any
    # and includes a rollback scheme if after patching one, the others get worse
    try_patching_multiple_at_same_time = True,
    ## if true, you metric will also have the logits from a run with no patching done
    # (useful for normalized logit diff)
    store_unpatched_logits = True,
)
```

`your_metric` should be a function that looks like this:

```python
def your_metric(data: ACDCEvalData):
    return # compute your metric

```

It's expected to return a tensor of size `[1]`.

`ACDCEvalData` contains three fields: `unpatched`, `patched`, and `corrupted`

(Note: `unpatched` will be None unless you set `store_unpatched_logits=True` in ACDCConfig)

Each of these are an `ACDCDataSubset` that contains fields useful for computing various metrics:

```python
@dataclass
class ACDCDataSubset:
    # from your ACDCDataset
    data: Float[torch.Tensor, 'batch n_ctx']
    last_token_position: Float[torch.Tensor, 'batch']
    correct: Float[torch.Tensor, 'batch n_correct']
    incorrect: Float[torch.Tensor, 'batch n_incorrect']
    constrain_to_answers: bool
    # the logits of the final token
    logits: Float[torch.Tensor, 'batch n_vocab']
    ## you can access all variables below just like any other variable,
    ## however because they can be expensive they are lazily loaded
    ## (only computed once you fetch them)
    # logits of correct answers
    correct_logits: Float[torch.Tensor, 'batch n_correct']
    # logits of incorrect answers
    incorrect_logits: Float[torch.Tensor, 'batch n_incorrect']
    # if constrain to answers=True, these will be normalized prs (softmax over only answer tokens (correct or incorrect))
    # if constrain_to_answers=False, these will be prs via softmax over all tokens in the vocab
    # prs of correct answers
    correct_logits: Float[torch.Tensor, 'batch n_correct']
    # prs of incorrect answers
    incorrect_logits: Float[torch.Tensor, 'batch n_incorrect']
    # True if the highest pr token is in correct
    top_is_correct: Float[torch.Tensor, 'batch']
```





