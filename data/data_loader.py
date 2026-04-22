from torch.utils.data import DataLoader
import torch
from rad_dino.utils import safetensors_to_state_dict
from transformers import pipeline
from extract_features import extract_features_rad_dino, extract_text_features
from models.imageModels import RadDINOFirst11Extractor
from models.textModels import UniModal_Text_ExtractFeatures
from utils import print_splits, stratified_split_multilabel, make_weighted_random_sampler, weightedClass, overSampling, print_20_test_image_ids
from data.dataset import CustomDataset, CachedFeatureDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data(data_grouped,mlb, training_mode, images_src,RadDino_src,RadDinoWeights, IMAGE_SIZE = 448, batch_size = 32, 
             wrs_mode = None, imagemodel = 0, textmodel = 0, DataAug = True, oversampler = False, freezeImage = False, freezeText = False):

    
    feature_extract_text = False
    feature_extract_image = False
    finetuning = False
    eval_bool = True
    if training_mode == 0 or training_mode == 2:
        if imagemodel != 3:
            if freezeImage:
                if imagemodel == 1:
                    feature_extract_image = True
                    image_encoder = torch.hub.load(RadDino_src, "dinov2_vitb14", source="local")
                    image_encoder = image_encoder
                    print(image_encoder)
                    backbone_state_dict = safetensors_to_state_dict(RadDinoWeights)

                    image_encoder.load_state_dict(backbone_state_dict, strict=True)
                elif imagemodel == 2:
                    feature_extract_image = True
                    eval_bool = False
                    image_encoder = pipeline(task="image-feature-extraction", model="microsoft/rad-dino-maira-2", pool=False, device=0)
                else: raise ValueError("There is no ResNet50 freeze mode added.")
            else:
                if imagemodel == 1:
                    finetuning = True
                    feature_extract_image = True
                    image_encoder = RadDINOFirst11Extractor(RadDino_src,RadDinoWeights, radDinoType = imagemodel)
                elif imagemodel == 2:
                    finetuning = True
                    feature_extract_image = True
                    image_encoder = RadDINOFirst11Extractor(RadDino_src,RadDinoWeights, radDinoType = imagemodel)
                    #print("RedDino MAIRA-2 fine-tuning was not added. (we will work with RedDino MAIRA-2 Freeze)")
                    #image_encoder = pipeline(task="image-feature-extraction", model="microsoft/rad-dino-maira-2", pool=False)
                else: print("FineTuning ResNet50")

    if freezeText:
        if training_mode == 1:
            feature_extract_text = True
            TextFeaturesextract_model = UniModal_Text_ExtractFeatures(encoder_type = textmodel).to(device)
        else:
            raise ValueError("The text backbone freeze can only be added for text-only training.")


    dataset_train = CustomDataset(data_grouped, images_src, training_mode, IMAGE_SIZE=IMAGE_SIZE, split="train", DataAug = DataAug)
    dataset_val   = CustomDataset(data_grouped, images_src, training_mode, IMAGE_SIZE=IMAGE_SIZE, split="val", DataAug = DataAug)
    dataset_test  = CustomDataset(data_grouped, images_src, training_mode, IMAGE_SIZE=IMAGE_SIZE, split="test", DataAug = DataAug)


    all_labels = dataset_train.get_labels_only()
    temp_idx, test_idx = stratified_split_multilabel(all_labels, n_splits=5, fold=0)
    test_dataset = torch.utils.data.Subset(dataset_test, test_idx)

    temp_dataset_full = all_labels[temp_idx]
    train_idx, val_idx = stratified_split_multilabel(temp_dataset_full, n_splits=4, fold=0)
    train_dataset = torch.utils.data.Subset(dataset_train, [temp_idx[i] for i in train_idx])
    val_dataset = torch.utils.data.Subset(dataset_val, [temp_idx[i] for i in val_idx])

    #Print the splits result
    print_splits(train_dataset, val_dataset, test_dataset, mlb.classes_)

    # ---- print 20 test samples with ImageID for Grad-CAM ----
    """print("\n=== 20 test samples (ImageID + labels) for Grad-CAM ===")
    selected_abs_idx = print_20_test_image_ids(
        test_idx=test_idx,
        df_grouped=data_grouped,
        all_labels=all_labels,
        label_names=mlb.classes_,
        max_samples=20
    )"""

    feat_dim = 768
    if oversampler:
        train_dataset = overSampling(temp_idx,train_idx, dataset_train)
        print_splits(train_dataset, val_dataset, test_dataset, mlb.classes_)

        if feature_extract_image:
            train_dataset_extractedFeatures, _ = extract_features_rad_dino(train_dataset,training_mode, image_encoder, finetuning, eval_bool)
            train_dataset = CachedFeatureDataset(train_dataset_extractedFeatures, training_mode)
        
        elif feature_extract_text:
            train_dataset_feats, _ = extract_text_features(train_dataset, TextFeaturesextract_model)
            train_dataset = CachedFeatureDataset(train_dataset_feats, training_mode)
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,shuffle=True)
        pos_weight = None

        pos_weight = None
    else:
        if wrs_mode:
            train_sampler = make_weighted_random_sampler(dataset_train, train_idx)

            with torch.no_grad():
                Y_train = dataset_train.get_labels_only()[train_idx]
                picked = list(train_sampler)               # indices 0..len(train_ds)-1
                approx_counts = Y_train[picked].sum(axis=0)
                print("Approx. train counts this epoch:", approx_counts.astype(int).tolist())

                Y_train_raw = torch.as_tensor(dataset_train.get_labels_only()[train_idx], dtype=torch.float32)
                P = Y_train_raw.sum(dim=0)                  # [C]
                N_total = Y_train_raw.shape[0]
                pos_weight = (N_total - P) / (P + 1e-6)     # (#negatives / #positives)

                # Optional: clamp to avoid huge gradients for ultra-rare classes
                pos_weight = torch.clamp(pos_weight, max=100.0)
                #print("pos_weight:", pos_weight.tolist())

            if feature_extract_image:
                train_dataset_extractedFeatures, _ = extract_features_rad_dino(train_dataset, training_mode, image_encoder, finetuning, eval_bool)
                train_dataset = CachedFeatureDataset(train_dataset_extractedFeatures, training_mode)
            
            elif feature_extract_text:
                train_dataset_feats, _ = extract_text_features(train_dataset, TextFeaturesextract_model)
                train_dataset = CachedFeatureDataset(train_dataset_feats, training_mode)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4,shuffle=True)
        
        else:

            if feature_extract_image:
                train_dataset_extractedFeatures, _ = extract_features_rad_dino(train_dataset,training_mode, image_encoder, finetuning, eval_bool)
                train_dataset = CachedFeatureDataset(train_dataset_extractedFeatures, training_mode)
            
            elif feature_extract_text:
                train_dataset_feats, _ = extract_text_features(train_dataset, TextFeaturesextract_model)
                train_dataset = CachedFeatureDataset(train_dataset_feats, training_mode)
                
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4,shuffle=True)
            #pos_weight = weightedClass(dataset_train,train_idx)


            abs_train_idx = [temp_idx[i] for i in train_idx]
            all_labels = dataset_train.get_labels_only()         # torch [N, C]
            pos_weight = weightedClass(all_labels, abs_train_idx)

    
    if feature_extract_image:
        val_dataset_extractedFeatures, _ = extract_features_rad_dino(val_dataset, training_mode, image_encoder, finetuning, eval_bool)
        test_dataset_extractedFeatures, feat_dim = extract_features_rad_dino(test_dataset, training_mode, image_encoder, finetuning, eval_bool)
        val_dataset = CachedFeatureDataset(val_dataset_extractedFeatures, training_mode)
        test_dataset = CachedFeatureDataset(test_dataset_extractedFeatures, training_mode)

    elif feature_extract_text:
        val_dataset_feats, _ = extract_text_features(val_dataset, TextFeaturesextract_model)
        val_dataset = CachedFeatureDataset(val_dataset_feats, training_mode)
        test_dataset_feats, feat_dim = extract_text_features(test_dataset, TextFeaturesextract_model)
        test_dataset = CachedFeatureDataset(test_dataset_feats, training_mode)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, pos_weight, feat_dim