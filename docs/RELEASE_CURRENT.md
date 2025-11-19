# Release 0.9.0

## 主要依赖库版本描述
- python 3.9 或者更高
- [torch-mlu] 2.7.1；
- [torch-npu] 2.7.1；
- pytorch与torch-mlu或torch-npu对应；
- [triton-x] 3.2.0.post9 或者更高；

## 重大变动
- 现在`GE`后端不再默认使能以下两个选项
  - `experimental_config.keep_inference_input_mutations`
  - `experimental_config.frozen_parameter`

  这是因为前者需要确保编译覆盖的算子，正确注册`mutates_args`；后者则在`regional compilaton`使用场景下会导致精度问题，我们决定交由使用者去控制这两个选项的开关，同时也保持和`vendor`默认设置的对齐；
  如果用户需要使能这两个开关，可以在`vendor_compiler_config`中增加`{"experimental_config": {"keep_inference_input_mutations": True, "frozen_parameter": True}}`这样的结构化字段来告知XpuGraph。

## 主要特性与改进
- 现在`Target.npu`后端支持了编译产物落盘缓存，以支持冷启动优化；xpu_graph设置里cache功能是默认打开的，但在之前的版本并不能生效，现在这个开关可以作用与`Target.npu`;
- 现在`XpuGraph`在`torch 2.7.1`版本上实现了`guard_filter`功能，使用方式为`torch.compile(backend=XpuGraph(), options={"guard_filter_fn": lambda guards: return [False for g in guards]})`，即返回每个guard的保留状态列表，用于过滤；如果你需要禁用全部guards，我们预定义了一个回调函数`from xpu_graph import skip_all_guards_unsafe`，只需将此函数传递给`guard_filter_fn`字段即可；

## Bug修复与其他改动
- 修复用户在传入`{"compiler": "ge"}`设置下，错误使用`AclGraph`而非`GE`的问题；
