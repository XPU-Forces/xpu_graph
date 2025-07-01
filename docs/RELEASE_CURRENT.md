# Release 0.1.0

## 主要依赖库版本描述
- python 3.9~11
- pytorch 2.5.1
- [torch-mlu]
- [torch-npu]

## 主要特性与改进
- 优化structure类pattern的组织形式，重构多个pattern

## 相关模块API的重大变动
-

## 弃用特性声明
-

## Bug修复与其他改动
- 迁移多个pattern到structure和common组，增强多后端复用
- 重构部分pattern，提升稳定性和可读性
- 在时间戳之后增加了进程与线程ID的打印，方便调试。
