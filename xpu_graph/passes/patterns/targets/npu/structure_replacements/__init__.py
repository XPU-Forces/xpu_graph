from .rms_norm_module import RMSNormModule
from .layer_norm_module import LayerNormModule


def get_structure_replacements(config):
    return {
        "FastRMSNorm": RMSNormModule,
        "FastLayerNorm": LayerNormModule,
    }
