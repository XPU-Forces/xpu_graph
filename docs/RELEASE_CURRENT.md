# Release 0.10.1

## 主要依赖库版本描述
- python 3.9 或者更高
- [torch-mlu] 2.7.1；
- [torch-npu] 2.7.1；
- pytorch与torch-mlu或torch-npu对应；
- [triton-x] 3.2.0.post9 或者更高；

## 主要特性与改进
- 增加`GraphRunner`用于提供基于`torch.nn.Module`的`NpuGraph`与`MluGraph`模块封装，使用样例：
  ```python
      import torch
      from xpu_graph import GraphRunner, Target
      model = torch.nn.Sequential(*[torch.nn.Linear(1024, 1024)] * 3).npu()
      input_tensor = torch.empty(1024, 1024).uniform_(10, 100).npu()

      golden = model(input_tensor)
      device_graph = GraphRunner[Target.npu](
          model,
          lambda input_tensor: input_tensor,
          lambda input_buffer, input_tensor: (input_buffer.copy_(input_tensor), True),
      )
      device_graph.capture(torch.empty_like(input_tensor).uniform_(1, 2))

      assert torch.allclose(device_graph(input_tensor), golden)
  ```

## Bug修复与其他改动
- 修复**CSE**相关pass错误移除有**副作用side-effect**节点的问题；
