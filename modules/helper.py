import torch

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def save_model(cust_model, name="dense_segm.pt"):
    return torch.save(cust_model.module.state_dict(), name)

def load_model(cust_model, model_dir="dense_segm.pt", map_location_device="cpu"):
    if map_location_device == "cpu":
        cust_model.load_state_dict(torch.load(model_dir, map_location=map_location_device))
    elif map_location_device == "gpu":
        cust_model.load_state_dict(torch.load(model_dir))
    cust_model.eval()
    return cust_model
