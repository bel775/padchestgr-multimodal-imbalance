from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
from torch.utils.data import WeightedRandomSampler
import torch
import torch.nn as nn
import pandas as pd
import multilabel_oversampling as mo
from sklearn.preprocessing import MultiLabelBinarizer

from clean_sentence import clean_sentence_label, clean_suspects_terms, remove_exclusive_terms

def preProcessData(data, label_count = 25, xlsx_path = "",):

    data_filtered = data[['ImageID', 'label', 'label_group','sentence_en']]
    data_filtered = data_filtered[data_filtered['label'] != "Normal"]
    print(data_filtered.head())

    top_labels = (
        data_filtered['label_group']
        .value_counts()
        .nlargest(label_count)     
        .index            
    )

    data_filtered = data_filtered[data_filtered['label_group'].isin(top_labels)]
    label_output_count = len(top_labels)

    print("Label Number Types: ", label_output_count)

    # Group labels by ImageID
    data_grouped = (
        data_filtered
        .groupby('ImageID')
        .agg({
            'label_group': lambda x: list(x),
            'sentence_en': lambda x: ' '.join(x.dropna())
        })
        .reset_index()
    )

    print("Remove label terms")
    data_grouped['sentence_en_clean'] = data_grouped.apply(clean_sentence_label, axis=1)

    #eliminate the empty sentences after removing the label terms
    data_grouped = data_grouped[data_grouped['sentence_en_clean'].str.strip() != '']
    data_grouped = data_grouped.reset_index(drop=True)

    if xlsx_path != "":
        print("Remove suspects terms")
        phrase_re = clean_suspects_terms(xlsx_path)
        data_grouped['sentence_en_clean_terms'] = data_grouped['sentence_en_clean'].apply(lambda txt: remove_exclusive_terms(txt, regex=phrase_re))

        # Replace the empty sentences with the previous filter after aplying the removing of the suspects terms
        data_grouped['final_sentence'] = np.where(
            data_grouped['sentence_en_clean_terms'].str.strip() == '',
            data_grouped['sentence_en_clean'],   
            data_grouped['sentence_en_clean_terms']  
        )
    
    #multi_hot
    mlb = MultiLabelBinarizer()
    multi_hot_labels = mlb.fit_transform(data_grouped['label_group'])

    data_grouped['multi_hot'] = list(multi_hot_labels)

    return data_grouped, mlb

def print_splits(train_dataset, val_dataset, test_dataset, classes):
    # Access original dataset
    full_dataset = train_dataset.dataset
    full_labels = full_dataset.get_labels_only()

    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    y_train = full_labels[train_indices]
    y_val = full_labels[val_indices]
    y_test = full_labels[test_indices]

    class_col_width = max(len(c) for c in classes) + 2
    header = f"{'Class':<{class_col_width}} | {'Train':>7} | {'Val':>7} | {'Test':>7}"
    print("\n" + header)
    print("-" * len(header))

    for idx, class_name in enumerate(classes):
        train_count = int(y_train[:, idx].sum())
        val_count = int(y_val[:, idx].sum())
        test_count = int(y_test[:, idx].sum())
        print(f"{class_name:<{class_col_width}} | {train_count:>7} | {val_count:>7} | {test_count:>7}")

def stratified_split_multilabel(labels, n_splits=5, fold=0):
    # Extract all labels from the dataset
    #labels = dataset.get_labels_only()

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i, (split1_idx, split2_idx) in enumerate(mskf.split(np.zeros(len(labels)), labels)):
        if i == fold:
            return split1_idx, split2_idx
        
def make_weighted_random_sampler(base_dataset, subset_indices):
    print("weighted random sampler")
    # labels: [N_subset, C] with {0,1}
    Y = base_dataset.get_labels_only()[subset_indices]
    Y = np.asarray(Y, dtype=np.float64)

    # per-class inverse frequency (clip to avoid div-by-zero)
    class_counts  = Y.sum(axis=0)                       # [C]
    class_weights = 1.0 / np.clip(class_counts, 1, None)

    # per-sample weight = average of its positive class weights
    pos_per_sample = Y.sum(axis=1)                      # [N_subset]
    sample_weights = (Y * class_weights).sum(axis=1) / np.maximum(pos_per_sample, 1)

    # handle rare all-zero rows (no positives)
    #if (pos_per_sample == 0).any():
    #    nz = sample_weights[pos_per_sample > 0]
    #    sample_weights[pos_per_sample == 0] = nz.mean() if nz.size else 1.0

    # (optional) scale so average weight ≈ 1; not required by the sampler
    #s = sample_weights.sum()
    #if s > 0:
    #    sample_weights = sample_weights * (len(sample_weights) / s)

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),   # one pass worth of draws
        replacement=True
    )

def weightedClass(all_labels, abs_train_idx):
    Y_train = all_labels[abs_train_idx]            # NumPy array
    pos_counts = Y_train.sum(axis=0)               # use axis, not dim
    N = Y_train.shape[0]
    pos_weight = (N - pos_counts) / np.maximum(pos_counts, 1)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

    return pos_weight

def overSampling(temp_idx,train_idx, dataset_train):
    train_orig_idx = [temp_idx[i] for i in train_idx]

    y_all = dataset_train.get_labels_only()              
    y_train = y_all[train_orig_idx]                      

    num_labels = y_train.shape[1]
    label_cols = [f"y{i+1}" for i in range(num_labels)]  # y1, y2, ...
    df_train = pd.DataFrame(y_train, columns=label_cols).astype(int)

    df_train.insert(0, "orig_idx", train_orig_idx)

    ml_oversampler = mo.MultilabelOversampler(number_of_adds=6000, number_of_tries=6000)
    df_train_os = ml_oversampler.fit(df_train, target_list=label_cols)

    new_train_idx = df_train_os["orig_idx"].tolist()
    train_dataset_os = torch.utils.data.Subset(dataset_train, new_train_idx)

    return train_dataset_os

def print_20_test_image_ids(test_idx, df_grouped, all_labels, label_names, max_samples=20):

    selected_abs_idx = []

    for abs_idx in test_idx:
        y = all_labels[abs_idx]
        pos = np.where(y == 1)[0]
        if len(pos) == 0:
            continue  # skip images with no labels

        img_id = df_grouped.iloc[abs_idx]['ImageID']
        class_names = [label_names[j] for j in pos]

        print(f"{len(selected_abs_idx):02d} | abs_idx={abs_idx} | ImageID={img_id} | labels={class_names}")

        selected_abs_idx.append(abs_idx)

        if len(selected_abs_idx) >= max_samples:
            break

    return selected_abs_idx

from models.textModels import UniModal_Text_Clasiffier, Linear_Classifier
from models.imageModels import UniModal_ResNet50, UniModal_RadDino, UniModal_RadDINOLastBlockClassifier, UniModal_radDino_LastBlock_PlusHead, UniModal_radDino_PlusHead
from models.multiModal import multiModel, MultiModal_RadDINOLastBlockClassifier

def get_model(textmodel,imagemodel,freezeImage,freezeText,RadDino_src,RadDinoWeights,
              Head_RadDinoWeights, num_classes=3, fusion_dim = 768, raddinoHead = False):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    head_Lr = 1e-3
    textBackbone_Lr = 2e-5
    imageBackboneResNet_Lr = 1e-4
    imageBackboneRadDino_Lr = 1e-5
    weight_decay = 1e-4

    if imagemodel == 3:
        if freezeText == False:
            model = UniModal_Text_Clasiffier(num_classes=num_classes, encoder_type = textmodel).to(device)
            trainable_params = [
                {"params": model.classifier.parameters(), "lr": head_Lr},
                {"params": (p for p in model.textencoder.parameters() if p.requires_grad), "lr": textBackbone_Lr},
            ]
        else:
            model = Linear_Classifier(fusion_dim, num_classes=num_classes).to(device)
            trainable_params = [
                {"params": model.classifier.parameters(), "lr": head_Lr}
            ]
        optimizer = torch.optim.AdamW(trainable_params, weight_decay=weight_decay)
        return model, optimizer
    elif textmodel == 4:
        if imagemodel == 0:
            model = UniModal_ResNet50(num_classes=num_classes).to(device)
            trainable_params = [
                {"params": model.classifier.parameters(), "lr": head_Lr},
                {"params": (p for p in model.resnet50.parameters() if p.requires_grad), "lr": imageBackboneResNet_Lr}
            ]
        else:
            if freezeImage == False:
                if raddinoHead:
                    model = UniModal_radDino_LastBlock_PlusHead(768,RadDino_src,RadDinoWeights,Head_RadDinoWeights,num_classes=num_classes).to(device)
                    trainable_params = [
                        {"params": model.last_block.parameters(), "lr": imageBackboneRadDino_Lr},
                        {"params": model.final_norm.parameters(), "lr": imageBackboneRadDino_Lr},
                        {"params": (p for p in model.rad_dino_head_gh.mlp.parameters() if p.requires_grad), "lr": imageBackboneRadDino_Lr},
                        {"params": (p for p in model.rad_dino_head_gh.last_layer.parameters() if p.requires_grad), "lr": head_Lr}
                        #{"params": model.classifier.parameters(), "lr": 1e-3}
                    ]
                else:
                    model = model = UniModal_RadDINOLastBlockClassifier(RadDino_src,RadDinoWeights,num_classes,in_dim= 768,radDinoType=imagemodel).to(device)
                    trainable_params = [
                        {"params": model.last_block.parameters(), "lr": imageBackboneRadDino_Lr}, #1e-3
                        {"params": model.final_norm.parameters(), "lr": imageBackboneRadDino_Lr},
                        {"params": model.classifier.parameters(), "lr": head_Lr}
                    ]
            else:
                if raddinoHead:
                    model = UniModal_radDino_PlusHead(768, Head_RadDinoWeights,num_classes=num_classes).to(device)
                    trainable_params = [
                        {"params": (p for p in model.rad_dino_head_gh.mlp.parameters() if p.requires_grad), "lr": imageBackboneRadDino_Lr},
                        {"params": (p for p in model.rad_dino_head_gh.last_layer.parameters() if p.requires_grad), "lr": head_Lr}
                        #{"params": model.classifier.parameters(), "lr": 1e-3}
                    ]
                else:                 
                    model = UniModal_RadDino(768,num_classes=num_classes).to(device)
                    trainable_params = [
                        {"params": model.classifier.parameters(), "lr": head_Lr}
                    ]
        optimizer = torch.optim.AdamW(trainable_params, weight_decay=weight_decay)
        return model, optimizer
    else:
        if freezeImage or imagemodel == 0:
            model = multiModel(768,imageModel=imagemodel, num_classes=num_classes,textModel = textmodel).to(device)
            trainable_params = []

            if model.imageModel == 0:
                trainable_params.append({"params": model.resnet50.parameters(), "lr": 1e-4})

            trainable_params.append({"params": model.textencoder.parameters(), "lr": 2e-5})
            trainable_params.append({"params": model.classifier.parameters(), "lr": 1e-3})
        else:
            model = MultiModal_RadDINOLastBlockClassifier(RadDino_src, RadDinoWeights, in_dim = 768, num_classes=num_classes, encoder_type = textmodel, weightsDiff = True).to(device)
            trainable_params = [
                {"params": (p for p in model.last_block.parameters() if p.requires_grad), "lr": imageBackboneRadDino_Lr},
                {"params": model.final_norm.parameters(), "lr": imageBackboneRadDino_Lr},
                {"params": (p for p in model.textencoder.parameters() if p.requires_grad), "lr": textBackbone_Lr},
                {"params": model.classifier.parameters(), "lr": head_Lr}
            ]
        
        optimizer = torch.optim.AdamW(trainable_params, weight_decay=weight_decay)
        return model, optimizer
    
def get_criterion(classWeightType, pos_weight):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if classWeightType:
        pos_weight = pos_weight.to(device)
        print("class weights Tensor: ", pos_weight)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("Class Weights On")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Class Weights Off")
    return criterion

