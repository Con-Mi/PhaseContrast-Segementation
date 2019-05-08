import torch
from LinkNetModel import DenseLinkModel
from helper import load_model


segm_model = DenseLinkModel(input_channels=1, num_classes=3)
segm_model = load_model(segm_model, model_dir="../../../binaries/")
dummyInput = torch.randn(1, 1, 512, 512)
tracedNet = torch.jit.trace(segm_model, example_inputs=dummyInput)
tracedNet.save("../../../jit_binaries/jit_denseNet_intersect_watersh_50epcs.pt")