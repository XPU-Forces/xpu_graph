from typing import Callable

import torch
import torch.fx as fx
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._pytree import tree_flatten

from xpu_graph.fx_utils import FxStage, has_storage
from xpu_graph.passes.optimizer import Optimizer

aten = torch.ops.aten

rand_ops = [
    aten.dropout,
    aten._fused_dropout,
    aten._standard_gamma,
    aten.bernoulli,
    aten.multinomial,
    aten.native_dropout,
    aten.normal,
    aten.poisson,
    aten.binomial,
    aten.rrelu,
    aten.rand_like,
    aten.rand,
    aten.randint,
    aten.randn,
    aten.randperm,
]


def get_aten_target(node: fx.Node) -> Callable:
    if hasattr(node.target, "overloadpacket"):
        return node.target.overloadpacket
    return node.target


# This method is adapt from https://github.com/pytorch/pytorch/blob/main/torch/_functorch/compile_utils.py#L42
# After the commit(https://github.com/pytorch/pytorch/commit/43c9b4e0e690a07fe69633de2bbf45e585a40e03), upstream‘s
# CSE is a little more strick, which will forbid all nodes who have same Storage with graph's outputs to enter CSE Pass.
# But actually, we only need to forbid the storage's producer and graph's outputs only.
# We fix this in xpu_graph temporarily, and we will commit a PR to Torch latter.
# return a new copy of torch.fx.graph.Graph with CSE applied to the input graph
def fx_graph_cse(fx_g: torch.fx.graph.Graph):
    new_graph = fx.Graph()
    env = {}  # map from node in the old graph to node in the new graph
    hash_env = {}  # map from hash to a node in the new graph
    token_map = {}  # map from hash to token

    from torch._inductor.pattern_matcher import compute_mutation_region_ids

    # Compatible with version torch==2.3.1
    try:
        from torch._inductor.pattern_matcher import same_mutation_regions
    except ImportError:

        def same_mutation_regions(a, b) -> bool:
            assert "mutation_region_id" in a.meta
            assert "mutation_region_id" in b.meta
            return a.meta["mutation_region_id"] == b.meta["mutation_region_id"]

    compute_mutation_region_ids(fx_g)  # type: ignore[arg-type]

    # Make a set of separate storages returned from the output, which will be preserved
    # when pruning.  This prevents us from deduplicating returned tensors which have
    # experienced identical operations, but are separate data structures in eager mode.
    output_node: fx.Node = next(iter(reversed(fx_g.nodes)))
    assert output_node.op == "output"

    output_storages = {
        StorageWeakRef(n.meta["val"].untyped_storage()) for n in output_node.all_input_nodes if has_storage(n)
    }

    output_storages_producers = set()
    for n in fx_g.nodes:
        if has_storage(n) and StorageWeakRef(n.meta["val"].untyped_storage()) in output_storages:
            output_storages_producers.add(n)
            output_storages.remove(StorageWeakRef(n.meta["val"].untyped_storage()))

    output_node_and_storage_producer = set(output_node.all_input_nodes) | output_storages_producers

    for n in fx_g.nodes:
        # The placeholder, output, and get_attr nodes are copied to the new graph without change
        # do not CSE away random operations
        if (
            n.is_impure()
            or get_aten_target(n) in rand_ops
            # aten.empty is non-deterministic, so don't CSE it.
            # Also, aten.empty is almost always fusible into its consumer,
            # so it's not worth CSEing.
            or get_aten_target(n) is aten.empty
            or n in output_node_and_storage_producer
        ):
            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
        else:  # n.op == 'call_function', should never see n.op == 'call_module' or 'call_method'
            # substitute args and kwargs members to their mapping in env if exists
            # specs can be used to reconstruct nested list/dictionaries
            def substitute(arg_list):
                arg_list, spec = tree_flatten(arg_list)
                for i in range(len(arg_list)):
                    v = arg_list[i]
                    if isinstance(v, torch.fx.node.Node) and v in env:
                        arg_list[i] = env[v]
                    if isinstance(v, (torch.SymBool, torch.SymInt, torch.SymFloat)):
                        arg_list[i] = v.node
                return tuple(arg_list), spec

            args, args_spec = substitute(n.args)
            kwargs, kwargs_spec = substitute(n.kwargs)

            # each token corresponds to a unique node
            # nodes with the same token can be substituted
            token = {
                "target": n.target,
                "args": args,
                "args_spec": args_spec,
                "kwargs": kwargs,
                "kwargs_spec": kwargs_spec,
            }

            # hash substituted args to a number, do not hash specs because specs are not hashable
            # We need to add type into hash to avoid situations like:
            # hash((primals_2, 1.0)) == hash((primals_2, 1))
            hash_arg = hash((tuple((a, type(a)) for a in args), tuple((a, type(a)) for a in kwargs)))
            hash_val = (n.target, hash_arg)

            # check if a node has a substitute and can be eliminated
            hash_val_in_hash_env = hash_val in hash_env
            overwrite_due_to_mutation = False
            if hash_val_in_hash_env and token_map[hash_val] == token:
                duplicate_n_prev = hash_env[hash_val]
                if same_mutation_regions(n, duplicate_n_prev):
                    env[n] = duplicate_n_prev
                    continue
                else:
                    # any futures duplicates should replace with n, not duplicate_n_prev
                    overwrite_due_to_mutation = True

            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
            if overwrite_due_to_mutation or not hash_val_in_hash_env:
                hash_env[hash_val] = new_node
                token_map[hash_val] = token

    return new_graph


class Cse(Optimizer):
    _support_stages = [
        FxStage.inference,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        cse_graph = fx_graph_cse(gm.graph)

        changed = len(cse_graph.nodes) != len(gm.graph.nodes)
        gm.graph = cse_graph

        return changed
