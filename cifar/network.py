from architectures import resnets

def wrn_16_4_network(model_dir,
    num_classes=10, load_weights=False):
  return resnets.cifar_wrn_16_4(model_dir=model_dir, 
    n_classes=num_classes, 
    load_weights=load_weights)

def wrn_40_4_network(model_dir,
    num_classes=10, load_weights=True):
  return resnets.cifar_wrn_40_4(model_dir=model_dir, 
    load_weights=load_weights, 
    n_classes=num_classes)
