import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoModel,AutoTokenizer
import torch

###############################################################################################
## TEXT EMBEDDING
###############################################################################################
class BertTokenizer_embedding(nn.Module):
    def __init__(self):
        super(BertTokenizer_embedding, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.output_dim = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.pooler_output
        return pooled

class BioBERT_Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
                "dmis-lab/biobert-base-cased-v1.1",
                revision="refs/pr/4",
                use_safetensors=True
            )
        self.output_dim = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.pooler_output

class CXR_BERT_Embedding(nn.Module):
    def __init__(self, cxr_type = "microsoft/BiomedVLP-CXR-BERT-general"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(cxr_type, use_safetensors=True, trust_remote_code=True)
        self.output_dim = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        
        last_hidden = out.last_hidden_state            # [B, L, H]
        mask = attention_mask.unsqueeze(-1).float()    # [B, L, 1]
        summed = (last_hidden * mask).sum(dim=1)       # [B, H]
        counts = mask.sum(dim=1).clamp(min=1e-9)       # [B, 1]
        return summed / counts
    
###############################################################################################
## TEXT MODELS
###############################################################################################
class UniModal_Text_Clasiffier(nn.Module):
    def __init__(self, num_classes=3, encoder_type = 0, max_length = 128):
        super().__init__()

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



        self.fusion_dim = self.textencoder.output_dim

        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, num_classes)
        )
        print("UniModal Text Clasiffier WIth 1 Linear")

    def forward(self, text):

        # Extract text features
        encoded = self.tokenizer(list(text), padding='max_length', truncation=True,
                            max_length=self.max_length, return_tensors='pt')
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        text_feats = self.textencoder(input_ids, attention_mask)

        return self.classifier(text_feats)

class UniModal_Text_ExtractFeatures(nn.Module):
    def __init__(self, encoder_type = 0, max_length = 128):
        super().__init__()

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

        self.fusion_dim = self.textencoder.output_dim

        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("Extract Text Features")

    def forward(self, text):

        # Extract text features
        encoded = self.tokenizer([text], padding='max_length', truncation=True, 
                            max_length=self.max_length, return_tensors='pt') #list(text)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        text_feats = self.textencoder(input_ids, attention_mask)

        return text_feats
    
class Linear_Classifier(nn.Module):
    def __init__(self, fusion_dim, num_classes=3):
        super().__init__()

        self.fusion_dim = fusion_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, num_classes)
        )
        print("1 Linear")
    
    def forward(self, text_feats):
        return self.classifier(text_feats)