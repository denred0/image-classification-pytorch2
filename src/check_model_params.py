import timm
from pprint import pprint

# model_names = timm.list_models(pretrained=True)

model_names = timm.list_models('*eff*')


pprint(model_names)

model = 'tf_efficientnetv2_l'
m = timm.create_model(model, pretrained=True)
pprint(m.default_cfg)

