import torch
from torchvision.transforms.functional import to_pil_image

def ensure_pil_uint8(img):
    # Accepts PIL or torch.Tensor
    if isinstance(img, torch.Tensor):
        # img: CxHxW
        if img.dtype.is_floating_point():
            img = (img.clamp(0, 1) * 255).to(torch.uint8)
        elif img.dtype != torch.uint8:
            img = img.to(torch.uint8)
        img = to_pil_image(img)
    return img

def ensure_pil_uint8_v2(img):
    # Accepts PIL or torch.Tensor
    if isinstance(img, torch.Tensor):
        # remove fake batch dim if present: 1xCxHxW -> CxHxW
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        if img.dtype.is_floating_point:
            img = (img.clamp(0, 1) * 255).to(torch.uint8)
        elif img.dtype != torch.uint8:
            img = img.to(torch.uint8)

        img = to_pil_image(img)
    return img

def output_to_Tensor(outputs):
    # coerce to a Tensor
    if hasattr(outputs, "last_hidden_state"):           
        cls = outputs.last_hidden_state
    elif isinstance(outputs, dict):                      
        cls = outputs.get("last_hidden_state", None)
        if cls is None:
            cls = next(v for v in outputs.values() if isinstance(v, torch.Tensor))
    elif isinstance(outputs, (tuple, list)):             
        cls = next(v for v in outputs if isinstance(v, torch.Tensor))
    elif isinstance(outputs, torch.Tensor):              
        cls = outputs
    else:
        raise TypeError(f"Unsupported encoder output: {type(outputs)}")
    return cls

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_features_rad_dino(dataset, training_mode, image_encoder, fineTuning, imagemodel):
    if imagemodel != 2:
        image_encoder.eval().to(device)

    all_features = []
    feat_dim = None

    with torch.no_grad():
        for batch in dataset:
            if not fineTuning and imagemodel == 2:
                image = batch['image_feat']
                image = ensure_pil_uint8_v2(image)
            else:
                if imagemodel == 2:
                    image = batch['image_feat'].unsqueeze(0)
                else:
                    image = batch['image_feat'].unsqueeze(0).to(device)
            label = batch['label'].to(device)

            if training_mode == 2:
                sentence = batch['sentence']


            cls_token = image_encoder(image) 
            if fineTuning:
                cls_token = output_to_Tensor(cls_token)
            
            if imagemodel == 2 and not fineTuning:
                cls_token = torch.tensor(cls_token, dtype=torch.float32)
                cls_token = cls_token[:, 0, :]

            #if feat_dim is None:
            #    print(cls_token.shape)
            #patch_embeddings = patch_tokens.mean(dim=[2, 3])                 

            #feats = torch.cat([cls_token, patch_embeddings], dim=1)           
            feats = cls_token.squeeze(0).cpu()    
            #feats = cls_token.cpu()                                  
            label = torch.as_tensor(label, dtype=torch.float32).cpu()        

            if feat_dim is None:
                print("Output Shape:", feats.shape)
                feat_dim = feats.shape[0]   # 1536
                print("Feature dim:", feat_dim)

            if training_mode == 0:
                all_features.append({
                    'image_feat': feats,        
                    'label': label              
                })
            else:
                all_features.append({
                    'image_feat': feats,        # CPU, (D,)
                    'sentence' : sentence,
                    'label': label              # CPU, (C,)
                })

    return all_features, feat_dim

def extract_text_features(dataset, text_encoder):
    text_encoder.eval().to(device)

    all_features = []
    feat_dim = None

    with torch.no_grad():
        for batch in dataset:
            label = batch['label'].to(device)
            sentence = batch['sentence']


            cls_token = text_encoder(sentence) 

            if feat_dim is None:
                print(cls_token.shape)
        
            feats = cls_token.squeeze(0).cpu()                                     
            label = torch.as_tensor(label, dtype=torch.float32).cpu()        

            if feat_dim is None:
                #feat_dim = text_encoder.fusion_dim
                feat_dim = feats.shape[0]   # 1536
                print("Feature dim:", feat_dim)

            all_features.append({
                'sentence': feats,        # CPU, (D,)
                'label': label              # CPU, (C,)
            })

    return all_features, feat_dim
