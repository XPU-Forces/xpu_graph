from .rms_norm_module import RMSNormModule


def get_structure_replacements(config):
    return {"CustomRMSNorm": RMSNormModule}
