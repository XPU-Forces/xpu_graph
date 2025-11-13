class XpuGraphRuntimeArtifact:
    def __init__(self, compiled_func):
        self._compiled_func = compiled_func

    def __call__(self, *args):
        if getattr(self._compiled_func, "_boxed_call", False):
            # Further reading: what is boxed_call? See discussion in : https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670
            return self._compiled_func(list(args))
        return self._compiled_func(*args)
