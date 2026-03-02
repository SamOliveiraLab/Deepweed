#!/usr/bin/env python3
"""Apply exact working pipeline (512,128), no norm, sigmoid 0.5 to notebook cell 10."""
import json

path = "Duckweed_Paper_Analysis.ipynb"
with open(path) as f:
    nb = json.load(f)

cell = nb["cells"][10]
raw = "".join(cell["source"])

# segment_boundary_model: replace single preprocessing block with DATASET_TYPE branch
a = """    original_shape = image_rgb.shape[:2]
    
    # Preprocess: resize to 256x256, normalize
    img = cv2.resize(image_rgb, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    input_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    
    # Predict
    with torch.no_grad():
        output = boundary_model(input_tensor)
    
    if DATASET_TYPE == "petri_dish":
        # 3-class: argmax → body class (1)
        pred = output.argmax(dim=1).squeeze().cpu().numpy()
        pred_full = cv2.resize(pred.astype(np.uint8), (original_shape[1], original_shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        body_mask = (pred_full == 1).astype(np.uint8)
    else:
        # 1-class: model outputs logits -> apply sigmoid then threshold
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_full = cv2.resize(pred, (original_shape[1], original_shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        body_mask = (pred_full > THRESHOLD).astype(np.uint8)"""

b = """    original_shape = image_rgb.shape[:2]
    if DATASET_TYPE == "petri_dish":
        img = cv2.resize(image_rgb, (256, 256))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        input_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    else:
        input_size = (512, 128)
        img = cv2.resize(image_rgb, (input_size[1], input_size[0]))
        img = img.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output = boundary_model(input_tensor)
    if DATASET_TYPE == "petri_dish":
        pred = output.argmax(dim=1).squeeze().cpu().numpy()
        pred_full = cv2.resize(pred.astype(np.uint8), (original_shape[1], original_shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        body_mask = (pred_full == 1).astype(np.uint8)
    else:
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        binary_mask_small = (pred > 0.5).astype(np.uint8)
        pred_full = cv2.resize(binary_mask_small, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        body_mask = (pred_full > 0.5).astype(np.uint8)"""

if a in raw:
    raw = raw.replace(a, b)
    lines = raw.split("\n")
    cell["source"] = [line + "\n" for line in lines[:-1]] + ([lines[-1] + "\n"] if lines[-1] else [])
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("Applied (512,128) + sigmoid 0.5 to segment_boundary_model")
else:
    print("Block A not found")
    if "Preprocess: resize to 256x256" in raw:
        print("(256,256) comment found but full block mismatch)")
    else:
        print("(256,256) comment not in cell)")
