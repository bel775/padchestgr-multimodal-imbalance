
import torch.nn as nn
from transformers import BertTokenizer,AutoTokenizer
import torch

from models.imageModels import ResNet50_Embedding, load_raddinoMaira1, _get_block_module, _get_final_norm, pool_tokens
from models.textModels import BertTokenizer_embedding, BioBERT_Embedding, CXR_BERT_Embedding


###################################################################################################
## General MultiModal
###################################################################################################
class multiModel(nn.Module):
    def __init__(self,img_feat_dim, imageModel=0, num_classes=3,textModel = 0, max_length = 128):
        super().__init__()

        self.imageModel = imageModel
        if imageModel == 0:
            self.resnet50 = ResNet50_Embedding()
            self.feat_dim = self.resnet50.output_dim
        else:
            self.feat_dim = img_feat_dim

        if textModel == 0:
            self.textencoder = BertTokenizer_embedding()
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            print("BertTokenizer")

        elif textModel == 1:
            self.textencoder = BioBERT_Embedding()
            self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
            print("BioBERT")
        
        else:
            cxr_type = ""
            if textModel == 2: cxr_type = "microsoft/BiomedVLP-CXR-BERT-general"
            else: cxr_type = "microsoft/BiomedVLP-CXR-BERT-specialized"
            self.textencoder = CXR_BERT_Embedding(cxr_type = cxr_type)
            self.tokenizer = AutoTokenizer.from_pretrained(cxr_type, use_safetensors=True, trust_remote_code=True)
            print(cxr_type)

        self.fusion_dim = self.feat_dim + self.textencoder.output_dim

        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.fusion_dim, num_classes)
        )
        print("MultiModal + 0.2 Dropout + 1 Linear")

    def forward(self, image, text):
        if self.imageModel == 0:
            image_feat = self.resnet50(image)
        else: image_feat = image

        encoded = self.tokenizer(list(text), padding='max_length', truncation=True,
                            max_length=self.max_length, return_tensors='pt')
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        text_feats = self.textencoder(input_ids, attention_mask)

        combined = torch.cat((image_feat, text_feats), dim=1)
        return self.classifier(combined)
    
###################################################################################################
## MultiModal with Last Block RadDino MAIRA-1
###################################################################################################
class MultiModal_RadDINOLastBlockClassifier(nn.Module):
    def __init__(self, RadDino_src, RadDinoWeights, in_dim = 768, num_classes=3, encoder_type = 0, max_length = 128, weightsDiff = False):
        super().__init__()

        # Image Encoder
        m = load_raddinoMaira1(RadDino_src, RadDinoWeights)
        self.last_block = _get_block_module(m, 11) 
        self.final_norm = _get_final_norm(m)

        self.weightsDiff = weightsDiff
        # Text Encoder
        if encoder_type == 0:
            self.textencoder = BertTokenizer_embedding()
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            print("BertTokenizer")

        elif encoder_type == 1:
            self.textencoder = BioBERT_Embedding()
            self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
            print("BioBERT")
        
        else:
            cxr_type = ""
            if encoder_type == 2: cxr_type = "microsoft/BiomedVLP-CXR-BERT-general"
            else: cxr_type = "microsoft/BiomedVLP-CXR-BERT-specialized"
            self.textencoder = CXR_BERT_Embedding(cxr_type = cxr_type)
            self.tokenizer = AutoTokenizer.from_pretrained(cxr_type,trust_remote_code=True, use_safetensors=True)
            print(cxr_type)



        self.fusion_dim = in_dim + self.textencoder.output_dim

        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.fusion_dim, num_classes)
        )
        print("1 Linear")


    def forward(self, image, text):
        # Extract image features
        x = self.last_block(image)  
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.final_norm(x)          
        if isinstance(x, (tuple, list)):
            x = x[0]
        image_feat = pool_tokens(x)    

        # Extract text features
        encoded = self.tokenizer(list(text), padding='max_length', truncation=True,
                            max_length=self.max_length, return_tensors='pt')
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        text_feats = self.textencoder(input_ids, attention_mask)

        #if self.weightsDiff:
        #    image_feat = F.normalize(image_feat, p=2, dim=-1)
        #    text_feats = F.normalize(text_feats, p=2, dim=-1)

        #    image_feat = 10 * image_feat
        #    text_feats = 0.1 * text_feats
        combined = torch.cat((image_feat, text_feats), dim=1)
        return self.classifier(combined)