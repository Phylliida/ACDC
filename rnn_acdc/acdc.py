import pickle
import torch
from dataclasses import dataclass, field
from jaxtyping import Float
from typing import List, Dict, Callable, Union, Tuple
from functools import partial
from collections import defaultdict
from functools import partial
from transformer_lens.hook_points import HookPoint
import networkx as nx
import graphviz
from tqdm.notebook import tqdm

def hook_was_added_by_hookpoint(hook):
    return "HookPoint." in str(repr(hook))

def clean_hooks(model):
    '''
    Removes all hooks from the model
    this is useful because sometimes if you spam ctrl-c too many times some hooks will stay around
    Due to interrupt between the time the hook is added to python
    but before python has a chance to add the handle to the list HookPoint holds onto
    this will make sure to even clean up those
    '''
    model.remove_all_hook_fns(including_permanent=True, level=None)
    model.reset_hooks(including_permanent=True, level=None)
    # extra stuff in case that wasn't everything
    for name, module in model.named_modules():
        for k, v in list(module._forward_hooks.items()):
            if hook_was_added_by_hookpoint(v):
                print("leftover hook", name, k, v.__name__, "removing")
                del module._forward_hooks[k]
            
        for k, v in list(module._forward_pre_hooks.items()):
            if hook_was_added_by_hookpoint(v):
                print("leftover hook", name, k, v.__name__, "removing")
                del module._forward_pre_hooks[k]
            
        for k, v in list(module._backward_hooks.items()):
            if hook_was_added_by_hookpoint(v):
                print("leftover hook", name, k, v.__name__, "removing")
                del module._backward_hooks[k]

def get_pad_token(tokenizer):
    '''
    Retreives the pad token id from the tokenizer
    '''
    return tokenizer.pad_token_id

def index_into(data, indices):
    '''
    given data that is [N,V] and indicies that are [N,K] with each index being an index into the V space
    this does what you'd want, it indexes them
    '''
    num_data, num_per_data = indices.size()
    # we want
    # [0,0,0,...,] num per data of these
    # [1,1,1,...,] num per data of these
    # ...
    # [num_data-1, num_data-1, ...]
    first_axis_index = torch.arange(num_data, dtype=torch.long).view(num_data, 1)*torch.ones([num_data, num_per_data], dtype=torch.long)
    # now we flatten it so it has an index for each term aligned with our indices
    first_axis_index = first_axis_index.flatten()

    second_axis_index = indices.flatten()
    # now we can just index, and then view back to our original shape
    return data[first_axis_index, second_axis_index].view(num_data, num_per_data)


class ACDCCachedEvalData:
    pass

def cached_property(f):
    '''
    Decorator that acts like property
    except it only calls it once,
    otherwise the value is cached inside self.data_cache
    (basically this lets you make lazy properties that only execute once called)
    '''
    property_name = f.__name__
    @property
    def wrapper(self, *args, **kwargs):
        if not hasattr(self.data_cache, property_name) or getattr(self.data_cache, property_name) is None:
            value = f(self, *args, **kwargs)
            setattr(self.data_cache, property_name, value)
            return value
        else:
            return getattr(self.data_cache, property_name)
    return wrapper

@dataclass
class ACDCDataSubset:
    logits: Float[torch.Tensor, 'batch n_ctx n_vocab']
    correct: Float[torch.Tensor, 'batch n_correct']
    incorrect: Float[torch.Tensor, 'batch n_incorrect']
    constrain_to_answers: bool

    data_cache: ACDCCachedEvalData = field(default_factory=lambda: ACDCCachedEvalData())

    @cached_property
    def correct_logits(self) -> Float[torch.Tensor, 'batch n_correct']:
        '''
        logits of the correct answers
        '''
        return index_into(self.logits[:,-1], self.correct)
    
    @cached_property
    def incorrect_logits(self) -> Float[torch.Tensor, 'batch n_incorrect']:
        '''
        logits of the incorrect answers
        '''
        return index_into(self.logits[:,-1], self.incorrect)

    @cached_property
    def correct_prs(self) -> Float[torch.Tensor, 'batch n_correct']:
        '''
        prs of the correct answers
        '''
        self.compute_data()
        return self.data_cache.correct_prs
    
    @cached_property
    def incorrect_prs(self) -> Float[torch.Tensor, 'batch n_incorrect']:
        '''
        prs of the incorrect answers
        '''
        self.compute_data()
        return self.data_cache.incorrect_prs
    
    @cached_property
    def top_is_correct(self) -> Float[torch.Tensor, 'batch']:
        '''
        1 if the highest pr answer is in the set of correct answers
        0 otherwise
        '''
        self.compute_data()
        return self.data_cache.top_is_correct

    def compute_data(self):
        '''
        Computes (correct_prs, incorrect_prs, has_correct)
        and stores them in the data cache
        '''
        if self.constrain_to_answers:
            correct_prs, incorrect_prs, top_is_correct = self.compute_constrained_data(
                correct_logits=self.correct_logits,
                incorrect_logits=self.incorrect_logits)
        else:
            correct_prs, incorrect_prs, top_is_correct = self.compute_unconstrained_data(
                logits=self.logits,
                correct=self.correct,
                incorrect=self.incorrect)
        self.data_cache.correct_prs = correct_prs
        self.data_cache.incorrect_prs = incorrect_prs
        self.data_cache.top_is_correct = top_is_correct
            
    def compute_constrained_data(self, correct_logits, incorrect_logits):
        '''
        Computes (correct_prs, incorrect_prs, top_is_correct)
        Where we constrain to only correct_logits, incorrect_logits
        '''
        n_data, n_correct = correct_logits.size()
        n_data, n_incorrect = incorrect_logits.size()
        # [n_data, n_correct + n_incorrect]
        combined_logits = torch.concatenate([correct_logits, incorrect_logits], dim=1)
        combined_prs = torch.softmax(combined_logits, dim=1)
        biggest = torch.argsort(-combined_prs, dim=1)
        # if biggest pr is in the correct, we are correct, otherwise, we are not
        top_is_correct = biggest[:,0] < n_correct
        correct_prs, incorrect_prs = combined_prs.split([n_correct, n_incorrect], dim=1)
        return (correct_prs, incorrect_prs, top_is_correct)

    def compute_unconstrained_data(self, logits, correct, incorrect):
        '''
        Computes (correct_prs, incorrect_prs, has_correct)
        Where we don't constrain to only correct_logits, incorrect_logits
        '''
        prs = torch.nn.functional.softmax(logits, dim=1)
        # [n_data, n_correct]
        correct_prs = index_into(prs, correct)
        # [n_data, n_incorrect]
        incorrect_prs = index_into(prs, incorrect)
        # [n_data, 1]
        top_tokens = torch.topk(logits, 1, dim=1).indices
        # [n_data, n_correct]
        is_correct = top_tokens == correct
        # [n_data]
        top_is_correct = torch.any(is_correct, dim=1) # if the top is equal to any of the entries in correct, return True
        return (correct_prs, incorrect_prs, top_is_correct)


@dataclass
class ACDCEvalData:
    logits_all_batches: Float[torch.Tensor, 'all_data_size ctx_len vocab_size']
    correct_all_batches: Float[torch.Tensor, 'all_data_size n_correct']
    incorrect_all_batches: Float[torch.Tensor, 'all_data_batch n_incorrect']
    batch_start: int = None
    batch_end: int = None
    constrain_to_answers: bool
    patched: ACDCDataSubset = field(default_factory=lambda: None)
    corrupted: ACDCDataSubset = field(default_factory=lambda: None)
    
    def set_batch(self, batch_start, batch_end):
        '''
        Initializes the data subsets
        '''
        self.batch_start = batch_start
        self.batch_end = batch_end
        del self.patched
        del self.corrupted
        self.patched = ACDCDataSubset(
            logits=self.logits_all_batches[::2][self.batch_start:self.batch_end],
            correct=self.correct_all_batches[::2][self.batch_start:self.batch_end],
            incorrect=self.incorrect_all_batches[::2][self.batch_start:self.batch_end],
            constrain_to_answers=self.constrain_to_answers,
        )
        self.corrupted = ACDCDataSubset(
            logits=self.logits_all_batches[1::2][self.batch_start:self.batch_end],
            correct=self.correct_all_batches[1::2][self.batch_start:self.batch_end],
            incorrect=self.incorrect_all_batches[1::2][self.batch_start:self.batch_end],
            constrain_to_answers=self.constrain_to_answers,
        )
    
def eval_acdc(model, data, correct, incorrect, metric, num_edges, constrain_to_answers, **kwargs):
    num_examples = correct.size()[0]
    
    logits = model(data, **kwargs)
    
    pad = get_pad_token(tokenizer=model.tokenizer)
    logits[:,pad] = -torch.inf # manually set pad pr to -inf logit because sometimes we need to pad num correct or num incorrect

    data = ACDCEvalData(
        logits_all_batches=logits,
        correct_all_batches=correct,
        incorrect_all_batches=incorrect,
        constrain_to_answers=constrain_to_answers)

    n_data = num_examples // 2 # every other one is corrupted
    metric_results = torch.zeros([num_edges], device=model.cfg.device)
    batch_size = n_data//num_edges
    for i, batch_start in enumerate(range(0, n_data, batch_size)):
        batch_end = batch_start + batch_size
        data.set_batch(batch_start=batch_start, batch_end=batch_end)
        metric_results[i] = metric(data=data)
    return metric_results

def accuracy_metric(data: ACDCEvalData):
    return torch.sum(data.patched.top_is_correct)/float(data.patched.top_is_correct.size()[0])

@dataclass
class ACDCConfig:
    input_node: str
    output_node: str
    metric: Callable
    thresh: Float = 0.0001
    rollback_thresh: Float = 0.004
    batch_size: int = 4
    batched: bool = True
    recursive: bool = True
    try_patching_multiple_at_same_time: bool = True
    layers: List[int] = field(default_factory=lambda: None)
    model_kwargs: Dict = field(default_factory=lambda: {})
    log_level: str = 'info' # can be 'debug', 'info', 'none', or 'all'
    auto_hide_unused_default_edges: bool = True # edges with None for input and output hook will always be visible, this helps reduce clutter by hiding them if they have no path through them to output or through them to input
    merge_positions_in_graph_display: bool = True # if you have seperate edges 1,2,3,5,6 going from node A to node B, this will visually display them as a single edge labeled 1-3,5-6

@dataclass
class ACDCDataset:
    data: Float[torch.Tensor, "batch context_len"]
    correct: Float[torch.Tensor, "batch num_correct_answers"]
    incorrect: Float[torch.Tensor, "batch num_incorrect_answers"]
    valid_data: Float[torch.Tensor, "batch context_len"]
    valid_correct: Float[torch.Tensor, "batch num_correct_answers"]
    valid_incorrect: Float[torch.Tensor, "batch num_incorrect_answers"]
    constrain_to_answers: bool

@dataclass
class Edge:
    input_node: str
    output_node: str
    # these can either be a single tuple (hook_name, hook_func)
    # or a list of tuples [(hook_name_1, hook_func_1), (hook_name_2, hook_func_2), etc.]
    input_hook: Union[List[tuple[str, Callable]],tuple[str, Callable]] = None
    output_hook: Union[List[tuple[str, Callable]],tuple[str, Callable]] = None
    label: str = None
    patching: bool = False
    checked: bool = False
    score_diff_when_patched: Float = False

    def is_default_edge(self):
        '''
        some edges are "default on" if they have no hooks
        they are "default on" because if they have no hooks we can't test them
        '''
        return self.input_hook is None and self.output_hook is None
    
    def get_hooks(self, **hook_args) -> List[Tuple[str, HookPoint]]:
        '''
        Gets a list of hooks from this edge, in the format
        [(hook_name_1, hook_func_1), (hook_name_2, hook_func_2), ...]
        Hooks are given hook_args as inputs,
        as well as the hook name(s) of the other side
        (so the input hook gets an output_hook_name param,
        and the output hook gets an input_hook_name param)
        '''
        hooks = []

        # validate, this helps prevent typo params from slipping through
        HOOK_ARGS = ['batch_start', 'batch_end']
        for k in hook_args:
            if not k in HOOK_ARGS:
                raise ValueError(f"Unknown hook key {k} valid values are {hook_args}, is there a typo?")

        # this is all a bit cursed to optionally handle being passed a list of hooks instead of a single hook
        def extract_hooks(hook, **kwargs):
            if hook is None: return []
            res = []
            if type(hook) is list:
                for h in hook:
                    res += extract_hooks(h, **kwargs)
            else:
                res.append((hook[0], partial(hook[1], **kwargs)))
            return res
        
        inputs = extract_hooks(self.input_hook, output_hook_name=None, **hook_args)
        outputs = extract_hooks(self.output_hook, input_hook_name=None, **hook_args)

        def extract_name(hooks_extracting_from):
            if len(hooks_extracting_from) == 0: return None
            
            res = [hook_name for (hook_name, hook_func) in hooks_extracting_from]
            if len(res) == 1: # if single hook, return it (not a list containing it)
                return res[0]
            else: # else we need to return list of names
                return res

        input_names = extract_name(inputs)
        output_names = extract_name(outputs)
        
        hooks += extract_hooks(self.input_hook, output_hook_name=output_names, **hook_args)
        hooks += extract_hooks(self.output_hook, input_hook_name=input_names, **hook_args)
        
        return hooks

    def __str__(self):
        has_label = self.label is not None and not self.label == 'None' and not len(self.label.strip()) == 0 
        label_str = self.label + " " if has_label else ""
        return f"edge {label_str} {self.input_node} -> {self.output_node} ({self.input_hook} -> {self.output_hook}))"

@dataclass
class Node:
    name: str
    incoming_edges: List[Edge] = field(default_factory=lambda: [])
    outgoing_edges: List[Edge] = field(default_factory=lambda: [])


def get_edges_to_check(edges : List[Edge], nodes : Dict[str, Node]):
    '''
    Returns edges that should be considered next for pruning by acdc
    This starts at the things going into the sink and works backward
    '''
    # goal is to find edges that have no path through them to some other unchecked edge
    # we could just topological sort and pick first edge that is unchecked
    # but we want to batch multiple edges
    # so what we do is:

    # 1. mark all edges as not ready to check
    for edge in edges:
        edge.ready_to_check = False
    
    # 2. start and mark any edges that go to sink nodes (nodes with no outgoing edges) as "ready_to_check"
    for node in nodes.values():
        if len(node.outgoing_edges) == 0: # if sink node  
            for edge in node.incoming_edges:
                edge.ready_to_check = True

    modified = True
    while modified:
        modified = False
        # 3. For every node:
        #      if all outgoing edges are ready_to_check and checked, we can mark all incoming edges as ready_to_check
        for node in nodes.values():
            all_outgoing_completed = all([outgoing_edge.checked and outgoing_edge.ready_to_check for outgoing_edge in node.outgoing_edges])
            if all_outgoing_completed:
                for edge in node.incoming_edges:
                    if not edge.ready_to_check:
                        modified = True
                        edge.ready_to_check = True
        # repeat until we have no more edges modified

    # now all edges that are ready_to_check and not checked are the ones on the frontier we want
    # they have no paths through them to other unchecked edges
    for edge in edges:
        if edge.ready_to_check and not edge.checked:
            yield edge

    ''' this simplified version works if edges only get checked starting from the end, but this doesn't hold when we have implicit edges
    for node_name, node in sorted(list(all_nodes.items())):
        # if we have checked all our outgoing edges (or we have none), we are safe to expand incoming edges
        all_outgoing_checked = all([outgoing_edge.checked for outgoing_edge in node.outgoing_edges])
        if all_outgoing_checked:
            for edge in node.incoming_edges:
                if not edge.checked:
                    yield edge
    '''


def get_nx_graph(edges : List[Edge], include_unchecked : bool = False) -> nx.DiGraph:
    '''
    Converts the edges into a networkx graph
    only edges that have checked == True and patching == False are included
    if include_unchecked=True, any edge that has checked == False is also included
    '''
    G = nx.DiGraph()
    labeldict = defaultdict(lambda: set())
    for edge in edges:
        if (edge.checked and not edge.patching) or (not edge.checked and include_unchecked):
            G.add_edge(edge.input_node, edge.output_node, label=edge.label)
    return G

def draw_nx_graph(edges : List[Edge]):
    '''
    Uses networkx to draw the graph
    '''
    import matplotlib.pyplot as plt
    G = get_nx_graph(edges=edges)
    nx.draw_networkx(G, with_labels=True, font_size=8)
    plt.show()


def merge_positions(positions : List[int]) -> str:
    '''
    [0,1,2,4,5,6,8] -> "0-2,4-6,8"
    '''
    if len(positions) == 0: return ""
    
    L = max(positions)+1
    
    is_visited = [False for _ in range(L)]
    for p in positions:
        is_visited[p] = True
    

    ranges = []
    earliest = min(positions)
    start_pos = earliest
    started_range = True
    for pos in range(earliest+1, L):
        if not is_visited[pos] and started_range:
            ranges.append((start_pos, pos-1))
            started_range = False
        elif is_visited[pos] and not started_range:
            start_pos = pos
            started_range = True
    if started_range:
        ranges.append((start_pos, L-1))
    out_strs = []
    for (start_pos, end_pos) in ranges:
        if start_pos == end_pos:
            out_strs.append(str(start_pos))
        else:
            out_strs.append(str(start_pos) + "-" + str(end_pos))
    out_str = ",".join(out_strs)
    return out_str

def has_path_to_terminal_nodes(
        input_node : str,
        output_node : str,
        edges : List[Edge],
        edges_to_check : List[Edge],
        include_unchecked=False,
        ignore_not_present=True
    ) -> List[Tuple[bool, bool]]:
    '''
    Returns a list of tuples (connected_to_input, connected_to_output)
    where the item i in that list corresponds to edges_to_check[i]
    and whether it has a path to input_node (connected_to_input) or
    output_node (connected_to_output)

    edges is used to construct the graph, which by default only considers edges
    that have checked=True and patching=False
    if include_unchecked=True, this graph will also include any edge with checked=False

    if ignore_not_present=False, exceptions will be thrown if the graph does not contain
    input_node or output_node
    '''
    import networkx as nx
    G = get_nx_graph(edges=edges, include_unchecked=include_unchecked)
    connected_to = []
    for edge in edges_to_check:
        connected_to_input = False
        try:
            to_input = nx.shortest_path(G, source=input_node, target=edge.input_node)
            connected_to_input = True
        except nx.NetworkXNoPath:
            pass
        except nx.NodeNotFound:
            if not ignore_not_present:
                raise ValueError(f"Graph does not have input node {input_node}")
        
        connected_to_output = False
        try:
            to_output = nx.shortest_path(G, source=edge.output_node, target=output_node)
            connected_to_output = True
        except nx.NetworkXNoPath:
            pass
        except nx.NodeNotFound:
            if not ignore_not_present:
                raise ValueError(f"Graph does not have output node {output_node}")
        connected_to.append((connected_to_input, connected_to_output))
    return connected_to
    
def get_graphviz_graph(cfg : ACDCConfig, edges : List[Edge]) -> graphviz.Digraph:
    '''
    Returns a graphviz.Digraph representing edges.
    If cfg.merge_positions is True, condense_positions will be used to visually simplify the graph a little
    If cfg.auto_hide_unused_default_edges is True, default edges (those with no input_hook and no output_hook) 
    will only be shown if they have a path to the input or a path to the output (this makes the graph cleaner)
    '''
    dot = graphviz.Digraph('result')
    node_connections = defaultdict(lambda: [])
    displaying_nodes = set()
    default_edges = []
    num_edges = 0
    def add_edge(edge):
        if not edge.label is None and len(edge.label) > 0 and 'None' != edge.label: # the 'None' happens when positions = [None]
            try:
                # this needs to be two lines because otherwise it'll populate the default dict on exception and add a ghost edge
                val = int(edge.label) 
                if cfg.merge_positions_in_graph_display:
                    node_connections[(edge.input_node, edge.output_node)].append(val)
                else:
                    dot.edge(edge.input_node, edge.output_node, label=edge.label + str(edge.score_diff_when_patched))
            except ValueError: # can't convert to int, just print the str
                dot.edge(edge.input_node, edge.output_node, label=edge.label + str(edge.score_diff_when_patched))
        else:
            dot.edge(edge.input_node, edge.output_node)
    
    for edge in edges:
        if edge.checked and not edge.patching:
            if cfg.auto_hide_unused_default_edges and edge.is_default_edge():
                default_edges.append(edge)
            else:
                add_edge(edge)
                num_edges += 1
                displaying_nodes.add(edge.input_node)
                displaying_nodes.add(edge.output_node)

    # only display these if they have path to input or path to output
    # otherwise it's a lot of clutter
    for edge, (connected_to_input, connected_to_output) in zip(default_edges, has_path_to_terminal_nodes(input_node=cfg.input_node, output_node=cfg.output_node, edges=edges, edges_to_check=default_edges)):
        if connected_to_input or connected_to_output:
            add_edge(edge)
            num_edges += 1

    # with things like [1,2,3,4,7,8] we turn them into 1-4,7-8
    for (input_node, output_node), positions in node_connections.items():
        positions = sorted(list(positions))
        dot.edge(input_node, output_node, label=merge_positions(positions))
        num_edges += 1

    if num_edges == 0:
        print(EMPTY_MESSAGE)
    
    return dot


EMPTY_MESSAGE = """Your graph has no edges, this probably means an edge was patched that caused input and output to become disconnected, resulting in a prune of the remaining edges (because they had no path from start to end)
We recommend resuming from the checkpoint before this occured, and observing the patches that caused it to be disconnected (try setting cfg.log_level='debug')
Most likely there is a bug in your patch code that causes these patches to not work properly, as a disconnected graph should function the same as completely patching the input,
and completely patching the input should result in a metric much lower than ACDC should allow. It is possible it slowly drifted that low, in this case you could turn off patching but it's likely your graph is meaningless at that point"""


def draw_graphviz_graph(cfg : ACDCConfig, edges : List[Edge]):
    '''
    Displays the graphviz graph corresponding to those edges,
    this assumes you are in a jupyter notebook and uses IPython.display
    '''
    from IPython.display import display
    display(get_graphviz_graph(cfg=cfg, edges=edges))


def get_currently_patched_edge_hooks(cfg: ACDCConfig, edges : List[Edge]):
    '''
    Get hooks for all edges that have patching=True and checked=True
    '''
    currently_patched_edge_hooks = []
    for edge in edges:
        if edge.patching and edge.checked:
            if cfg.batched: currently_patched_edge_hooks += edge.get_hooks(batch_start=0, batch_end=None) # None in a slice means grab everything
            else: currently_patched_edge_hooks += edge.get_hooks()
        elif edge.patching and not edge.checked:
            raise ValueError(f"Patching edge but not checked, what u doin, with edge {edge}")
    return currently_patched_edge_hooks

def load_checkpoint(path : str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data['cfg'], data['edges']

def save_checkpoint(cfg : ACDCConfig, edges : List[Edge], path : str):
    with open(path, 'wb') as f:
        pickle.dump({'cfg': cfg, 'edges': edges}, f)

def prune_edges(input_node : str, output_node : str, edges : List[Edge]):
    '''
    prunes edges that don't have a possible path (through checked=True and patching=False, or checked=False) from input_node -> them -> output_node
    Basically, if the edge couldn't possibly
    '''
    modified = True
    while modified:
        modified = False
        
        edges_to_check = [edge for edge in edges if edge.checked and not edge.patching] # only potentially prune edges that are part of our graph

        for edge, (connected_to_input, connected_to_output) in zip(edges_to_check, has_path_to_terminal_nodes(input_node=input_node, output_node=output_node, edges=edges, edges_to_check=edges_to_check, include_unchecked=True, ignore_not_present=False)):
            pruning = False
            if not connected_to_input:
                print(f"pruning edge {edge} because it doesn't have a path from {input_node} -> {edge.input_node}")
                pruning = True
            if not connected_to_output:
                print(f"pruning edge {edge} because it doesn't have a path from {edge.output_node} -> {output_node}")
                pruning = True
            if pruning:
                edge.checked = True
                edge.patching = True
                modified = True

def wrap_run_with_hooks(model, fwd_hooks, bwd_hooks=[], **kwargs):
    '''
    Makes a fake object that acts like model
    but when you call it it'll actually call run_with_hooks with the provided hooks
    '''
    def wrapper(input, fwd_hooks, bwd_hooks):
        #print(f"running model with {len(fwd_hooks)} fwd hooks and {len(bwd_hooks)} bwd hooks")
        return model.run_with_hooks(input, fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks, **kwargs)
    wrapper_with_hooks = partial(wrapper, fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks)
    wrapper_with_hooks.tokenizer = model.tokenizer
    wrapper_with_hooks.cfg = model.cfg
    return wrapper_with_hooks

LOG_LEVEL_FULL = 'full'
LOG_LEVEL_DEBUG = 'debug'
LOG_LEVEL_INFO = 'info'
INFO_LEVELS = [LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_FULL]

def run_acdc(model, cfg : ACDCConfig, data : ACDCDataset, edges : List[Edge]):
    '''
    Runs ACDC
    '''
    # this is useful because sometimes if you spam ctrl-c too many times some hooks will stay around
    clean_hooks(model)

    if not cfg.batched and cfg.batch_size > 1:
        raise ValueError(f"cfg.batched is False, so cfg.batch_size needs to be 1, instead it is {cfg.batch_size}")
    
    def get_baseline_score(edges):
        currently_patched_edge_hooks = get_currently_patched_edge_hooks(cfg=cfg, edges=edges)
        return eval_acdc(
            model=wrap_run_with_hooks(model=model, fwd_hooks=currently_patched_edge_hooks, **cfg.model_kwargs),
            data=data.data,
            correct=data.correct,
            incorrect=data.incorrect,
            metric=cfg.metric,
            num_edges=1,
            constrain_to_answers=data.constrain_to_answers)[0].item()
    
    # if we aren't given a set of limited layers, run on all layers
    limited_layers = cfg.layers
    if limited_layers is None:
        limited_layers = list(range(model.cfg.n_layers))
    
    # get all the nodes
    all_node_names = set()
    for edge in edges:
        all_node_names.add(edge.input_node)
        all_node_names.add(edge.output_node)
    if cfg.log_level in INFO_LEVELS:
        print("all nodes:", sorted(list(all_node_names)))
    all_nodes = dict([(name, Node(name=name)) for name in sorted(list(all_node_names))])
    for edge in edges:
        all_nodes[edge.input_node].outgoing_edges.append(edge)
        all_nodes[edge.output_node].incoming_edges.append(edge)
    
    # todo: validate all nodes have path to output
    if cfg.log_level == LOG_LEVEL_FULL:
        print("listing all nodes and edges:")
        for name, node in all_nodes.items():
            print("node", name)
            print("incoming")
            for edge in node.incoming_edges:
                print("   ", edge)
            print("outgoing")
            for edge in node.outgoing_edges:
                print(edge)
    
    # edges without hooks are always on
    for edge in edges:
        if edge.is_default_edge():
            edge.checked = True
            edge.patching = False
        
    iters = 0
    while len(list(get_edges_to_check(edges=edges, nodes=all_nodes))) > 0:
    
        # get edges that are next to check (because they don't have any edges after them that haven't been checked)
        edges_to_check = list(get_edges_to_check(edges=edges, nodes=all_nodes))
    
        def print_stats(edges):
            # get hooks for currently patched edges
            num_patching, num_keeping = 0, 0
            for edge in edges:
                if edge.patching:
                    num_patching += 1
                elif edge.checked:
                    num_keeping += 1
            
            if cfg.log_level in INFO_LEVELS:
                print(f"patching {num_patching} edges, keeping {num_keeping} edges, {len(edges)-num_patching-num_keeping} remain")
        
        print_stats(edges)
        out_scores = torch.zeros([len(edges_to_check)], device=model.cfg.device)

        # only join edges into a single forward pass if we are using the recursive trick
        # we also check to see if it's even worth doing (if it only takes a few forward passes
        # to try them all then why bother)
        FWD_PASSES_CONSIDERED_TOO_MANY = 3
        if cfg.recursive and len(edges_to_check) > cfg.batch_size * FWD_PASSES_CONSIDERED_TOO_MANY:
            edge_sets_to_process = [edges_to_check]
        else:
            # don't do the recursive binary pullapart thing, just one at a time
            edge_sets_to_process = [[edge] for edge in edges_to_check]
        edge_sets_to_patch = []
        edge_sets_to_keep = []
        while len(edge_sets_to_process) > 0:
            
            print_stats(edges)
            baseline_score = get_baseline_score(edges)
            
            new_edge_sets_to_process = []
            # if we aren't trying to patch multiple at the same time, only process a single data point, store the rest for later
            if not cfg.try_patching_multiple_at_same_time:
                new_edge_sets_to_process = edge_sets_to_process[1:]
                edge_sets_to_process = [edge_sets_to_process[0]]
            
            if cfg.log_level in INFO_LEVELS:
                print(f"baseline score {baseline_score}")
            if cfg.log_level in INFO_LEVELS:
                print(f"{[len(e) for e in edge_sets_to_process+new_edge_sets_to_process]} sized edge sets remaining")
            
            torch.set_grad_enabled(False)
            edge_sets_scores = torch.zeros([len(edge_sets_to_process)], device=model.cfg.device)
            batches = range(0, len(edge_sets_to_process), cfg.batch_size)
            #print(f"batches {batches}")
            if len(batches) > 2:
                batches = tqdm(list(batches))
            for batch_start in batches:
                torch.cuda.empty_cache()
                
                batch_end = min(batch_start+cfg.batch_size, len(edge_sets_to_process))
                batch_size = batch_end-batch_start
                
                # stack data for each edge
                def stack_data_for_each_edge(data, n_edges):
                    if n_edges == 1: return data
                    else: return torch.cat([data]*n_edges, dim=0)
                
                pbatched_data_edges = stack_data_for_each_edge(data.data, n_edges=batch_size)
                pbatched_correct_edges = stack_data_for_each_edge(data.correct, n_edges=batch_size)
                pbatched_incorrect_edges = stack_data_for_each_edge(data.incorrect, n_edges=batch_size)
                
                data_size = data.data.size()[0]

                batch_edge_sets = edge_sets_to_process[batch_start:batch_end]
                
                # add hooks for new edges we are testing
                # each edge set writes to a different piece of the batch
                hooks = get_currently_patched_edge_hooks(cfg=cfg, edges=edges)
                for i, edge_set in enumerate(batch_edge_sets):
                    batch_data_start = i*data_size
                    batch_data_end = batch_data_start+data_size
                    for patching_edge in edge_set:
                        baseline_hooks_len = len(hooks)
                        if cfg.batched: hooks += patching_edge.get_hooks(batch_start=batch_data_start, batch_end=batch_data_end)
                        else: hooks += patching_edge.get_hooks()
                        num_hooks_added = len(hooks)-baseline_hooks_len
                # get scores
                scores = eval_acdc(
                    model=wrap_run_with_hooks(model=model, fwd_hooks=hooks, **cfg.model_kwargs),
                    data=pbatched_data_edges,
                    correct=pbatched_correct_edges,
                    incorrect=pbatched_incorrect_edges,
                    metric=cfg.metric,
                    num_edges=batch_size,
                    constrain_to_answers=data.constrain_to_answers)
        
                edge_sets_scores[batch_start:batch_end] = scores
            if cfg.log_level in INFO_LEVELS:
                print("got scores", edge_sets_scores)
            edge_sets_to_patch = []
            for score, edge_set in zip(edge_sets_scores, edge_sets_to_process):
                score_lost_by_edge_set = baseline_score - score
                # if all those edges are less than thresh, we can patch them all
                if score_lost_by_edge_set <= cfg.thresh:
                    edge_sets_to_patch.append((score, score_lost_by_edge_set, edge_set))
                # otherwise, we need to split it apart
                else:
                    # if it's a single edge, we keep it
                    if len(edge_set) == 1:
                        if cfg.log_level in INFO_LEVELS:
                            print(f"keeping edge {edge_set[0]} with score {score} which has diff {score_lost_by_edge_set} > {cfg.thresh}")
                        for edge in edge_set:
                            edge.patching = False
                            edge.checked = True
                            edge.score_diff_when_patched = score_lost_by_edge_set
                    # otherwise, split into two edge sets
                    else:
                        if cfg.log_level in INFO_LEVELS:
                            print(f"splitting {len(edge_set)} edges into two edge sets, as they have score {score} which has diff {score_lost_by_edge_set} > {cfg.thresh}")
                        new_edge_sets_to_process.append(edge_set[:len(edge_set)//2])
                        new_edge_sets_to_process.append(edge_set[len(edge_set)//2:])
    
            # rollback, sometimes even though individually they cause no problem, patching them at same time causes a problem
            # this looks to see if that happened, if so, we need to find the smallest set that patching together doesn't cause a problem
            # we just do a greedy thing and consider half, then a fourth, then eigth, etc.
            if len(edge_sets_to_patch) > 0:
                # if we rollback we keep the earlier ones patched, make them the most promising
                edge_sets_to_patch.sort(key=lambda x: x[1])
                
                for score, score_lost_by_edge_set, edge_set in edge_sets_to_patch:
                    print(f"doing rollback test with score {score} and score_lost_by_edge_set {score_lost_by_edge_set} with edge set of size {len(edge_set)}")
                    for edge in edge_set:
                        edge.patching = True
                        edge.checked = True
                prev_baseline = baseline_score
                baseline_score = get_baseline_score(edges)
                metric_lost = prev_baseline - baseline_score
                pivot = len(edge_sets_to_patch)
                if metric_lost > cfg.rollback_thresh:
                    if cfg.log_level in INFO_LEVELS:
                        print(f"lost {metric_lost} when patching edge sets of size {[len(es) for (_,_,es) in edge_sets_to_patch]} finding rollback pivot")
                    pivot = pivot//2
    
                    while pivot > 0:
                        for i, (score, score_lost_by_edge_set, edge_set) in enumerate(edge_sets_to_patch):
                            patching = i <= pivot
                            for edge in edge_set:
                                edge.patching = patching
                                edge.checked = patching
                        baseline_score = get_baseline_score(edges)
                        metric_lost = prev_baseline - baseline_score
                        if metric_lost > cfg.rollback_thresh:
                            if cfg.log_level in INFO_LEVELS:
                                print(f"lost too much ({metric_lost}) when patching 0-{pivot} only (out of {len(edge_sets_to_patch)} total), rolling back more")
                            pivot = pivot//2
                        else:
                            if cfg.log_level in INFO_LEVELS:
                                print(f"rollback succeeded (lost {metric_lost}) when patching 0-{pivot} only (out of {len(edge_sets_to_patch)} total)")
                            break
                print(f"rolling back with pivot {pivot}")
                for i, (score, score_lost_by_edge_set, edge_set) in enumerate(edge_sets_to_patch):
                    patching = i <= pivot
                    for edge in edge_set:
                        edge.patching = patching
                        edge.checked = patching
                    if patching:
                        if cfg.log_level in INFO_LEVELS:
                            print(f"patching {len(edge_set)} edges with score {score} with diff of {score_lost_by_edge_set}")
                            if pivot > 0:
                                print(f"but patching with others at the same time, overall they have score {baseline_score} with diff of {metric_lost}")
                        if cfg.log_level == LOG_LEVEL_DEBUG:
                            for edge in edge_set:
                                if cfg.log_level in INFO_LEVELS:
                                    print(f"    patching edge {edge}")
                    else:
                        # these edge sets failed, add them back to the pile
                        new_edge_sets_to_process.append(edge_set)
                
            edge_sets_to_process = new_edge_sets_to_process
        if cfg.log_level in INFO_LEVELS:
            print("finished these edge sets")
        for edge_set in edge_sets_to_keep:
            for edge in edge_set:
                edge.patching = False
                edge.checked = True

        # prune edges that could not ever be on a path from input to output (because we have patched away all possible ways they could connect)
        prune_edges(input_node=cfg.input_node, output_node=cfg.output_node, edges=edges)
        
        hooks = get_currently_patched_edge_hooks(cfg=cfg, edges=edges)
        valid_score = eval_acdc(
                        model=wrap_run_with_hooks(model=model, fwd_hooks=hooks, **cfg.model_kwargs),
                        data=data.valid_data,
                        correct=data.valid_correct,
                        incorrect=data.valid_incorrect,
                        metric=cfg.metric,
                        num_edges=1,
                        constrain_to_answers=data.constrain_to_answers)
        if cfg.log_level in INFO_LEVELS:
            print(f"valid score {valid_score}")
            draw_graphviz_graph(cfg=cfg, edges=edges)
        ckpt_path = f"checkpoint {iters}.pkl"
        save_checkpoint(cfg=cfg, edges=edges, path=ckpt_path)
        if cfg.log_level in INFO_LEVELS:
            print(f"saved to checkpoint {ckpt_path}")
        iters += 1

    if cfg.log_level in INFO_LEVELS:
        print("final output:")
        draw_graphviz_graph(cfg=cfg, edges=edges)
    
    patching_hooks = get_currently_patched_edge_hooks(cfg=cfg, edges=edges)
    
    baseline_score = eval_acdc(
                model=wrap_run_with_hooks(model=model, fwd_hooks=patching_hooks, **cfg.model_kwargs),
                data=data.data,
                correct=data.correct,
                incorrect=data.incorrect,
                metric=cfg.metric,
                num_edges=1,
                constrain_to_answers=data.constrain_to_answers)[0].item()
    
    if cfg.log_level in INFO_LEVELS:
        print("final score", baseline_score)
    
    ckpt_path = f"checkpoint final.pkl"
    save_checkpoint(cfg=cfg, edges=edges, path=ckpt_path)
    if cfg.log_level in INFO_LEVELS:
        print(f"saved to checkpoint {ckpt_path}")

    return edges