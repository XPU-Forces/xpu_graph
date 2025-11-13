# Release 0.9.0

## 主要依赖库版本描述
- python 3.9 或者更高
- [torch-mlu] 2.7.1；
- [torch-npu] 2.7.1；
- pytorch与torch-mlu或torch-npu对应；
- [triton-x] 3.2.0 或者更高；

## 重大变动
- 新增Fallback Legacy Dispatch配置，默认开启。 #433
    * 开启时，使用现有的 dispatch & compile 流程；
    * 关闭时，则使用新版本的 dispatch & compile 流程。
-

## 主要特性与改进
-

## Bug修复与其他改动
-
