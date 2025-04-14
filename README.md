# xpu_graph
![arch](./doc/xpu_graph_arch.png)
XPU_GRAPH is a graph compiler based on torch [Fx graph](https://pytorch.org/docs/stable/fx.html) and [Aten IR](https://pytorch.org/docs/stable/torch.compiler_ir.html).

Here are some features of XPU_GRAPH:
* General graph optimizations: CSE, DCE, Op folding, Constant folding, and more aggresive constant propagation.
* Vendor custom op conveter: convert less efficient ops (who will often cause a lot of memory access) to custom fused op.
* Structure patterns: XPU_GRAPH abstracts common structural patterns, allowing users to implement the corresponding target structure. XPU_GRAPH will then convert the specified structure into the user-defined format.
* Backend compiler integration: XPU_GRAPH is a fx-graph-in and fx-graph-out graph compiler, so it is compatible with other fx graph compiler, like Inductor and GE.


## Environment requirements
```bash
python -m pip install -r requirements.txt
```

## Usage
Install xpu_graph
```bash
python -m pip install .
```

Then you can use xpu_graph with [PT2 compile](https://pytorch.org/docs/stable/generated/torch.compile.html) like this:

```python
def foo(x, y):
    z = x + y
    another_z = x + y
    return z, another_z

import torch
from xpu_graph.compiler import XpuGraph
compiled_foo = torch.compile(foo, backend=XpuGraph(XpuGraphConfig(False)))
compiled_foo(torch.randn(10), torch.randn(10))

```
