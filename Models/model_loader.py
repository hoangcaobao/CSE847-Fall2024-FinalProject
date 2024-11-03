from Models.model import Conv2DModel, ResNet50
def model_loader(cfg_proj, cfg_m):
    if cfg_proj.model == "simpleCNN":
        model = Conv2DModel(dim_out=cfg_m.data.dim_out, in_channels=cfg_m.data.in_channels, dataset_name=cfg_proj.dataset_name)
    if cfg_proj.model == "resnet50":
        model = ResNet50(dim_out=cfg_m.data.dim_out, in_channels=cfg_m.data.in_channels, dataset_name=cfg_proj.dataset_name)
    return model