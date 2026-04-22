import os
from data.data_loader import get_data
from train import train_model
from evaluation import evaluate_model
import pandas as pd
import argparse
from utils import get_model, get_criterion, preProcessData
dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def main(wrs_mode, classWeighted, dataAug, oversampling, imagemodel, raddinoHead, textmodel, freezeImage,freezeText, label_count):
    print("Start the Main ...")

    data = pd.read_csv(os.path.join(dir,"padchestgr-multimodal-imbalance/master_table.csv"))

    images_src = os.path.join(dir,'padchestgr-multimodal-imbalance/images/PadChest_GR')
    RadDino_src = os.path.join(dir,'dinov2_src')
    RadDinoWeights = os.path.join(dir,'padchestgr-multimodal-imbalance/backbones/backbone_compatible.safetensors')
    #RadDinoWeights = os.path.join(dir,'models/radDinoMaria2.safetensors')

    Head_RadDinoWeights = os.path.join(dir,'padchestgr-multimodal-imbalance/backbones/dino_head.safetensors')

    supects_terms_path = os.path.join(dir,"padchestgr-multimodal-imbalance/terminos_sospechosos_coincidencia_lexica.xlsx")
    #model = radDino_Head(768,Head_RadDinoWeights, 25).to('cuda')
    #print(model)
    #model = None

    if textmodel == 4 and imagemodel == 3:
        raise ValueError("Invalid configuration: both text and image training are disabled. Select at least one model.")


    wrs_str = "_WRS" if wrs_mode else ""
    cw_str = "_CW" if classWeighted else ""
    dataAug_str = "_DataAug" if dataAug else ""
    os_str = "_OS" if oversampling else ""
    ftImage_str = "Freeze" if freezeImage else ""
    ftText_str = "Freeze" if freezeText else ""

    textmodel_str = ""
    if textmodel == 0: textmodel_str = "_BertTokenizer"
    elif textmodel == 1: textmodel_str = "_BioBert"
    elif textmodel == 2: textmodel_str = "_CXR-Bert"
    elif textmodel == 3: textmodel_str = "_CXRBert_Especialized"
    else: textmodel_str = ""

    training_mode = 0 # 0 = Image-Only, 1 = Text-Only, 2 = MultiModal
    imagemodel_str = ""
    if imagemodel == 0: imagemodel_str = "_ResNet50"
    elif imagemodel == 1: imagemodel_str = "_RadDino_MAIRA1"
    elif imagemodel == 2: imagemodel_str = "_RadDino_MAIRA2"
    else: imagemodel_str = ""

    if textmodel == 4:
        training_mode = 0
        print("Image unimodal of type:", imagemodel_str)
    elif imagemodel == 3:
        training_mode = 1
        print("Text unimodal of type:", textmodel_str)
    else:
        training_mode = 2
        print("Multimodal of types:", textmodel_str, "and", imagemodel_str)

    batch_size = 32
    dataset, mlb = preProcessData(data, label_count = label_count, xlsx_path = supects_terms_path)
    train_loader, val_loader, test_loader, pos_weight, feat_dim = get_data(dataset, mlb, training_mode, images_src,RadDino_src,RadDinoWeights, IMAGE_SIZE = 518, 
                                                                                      wrs_mode= wrs_mode, imagemodel = imagemodel, textmodel = textmodel,
                                                                                      DataAug = dataAug, oversampler = oversampling, 
                                                                                      freezeImage = freezeImage, freezeText = freezeText)
    
    save_loss_path = os.path.join(dir,f'padchestgr-multimodal-imbalance/graphs/loss_curve{imagemodel_str}{ftImage_str}{textmodel_str}{ftText_str}{wrs_str}{cw_str}{dataAug_str}{os_str}.png')

    model,optimizer = get_model(textmodel,imagemodel,freezeImage,freezeText,RadDino_src,
                                RadDinoWeights,Head_RadDinoWeights,num_classes=label_count, 
                                fusion_dim = feat_dim, raddinoHead = raddinoHead)
    criterion = get_criterion(classWeighted, pos_weight)
    model = train_model(model,optimizer,criterion, training_mode, train_loader, val_loader,freezeText, save_loss_path)

    #torch.save(model.state_dict(), os.path.join(dir,'save_models/RadDinoMAIRA1FT_5label.pth'))

    evaluate_model(model, test_loader,training_mode,freezeText, eval_test = True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run main training script with configurable options.")

    ## Balance Strategy
    parser.add_argument("--wrs", action="store_true", help="Enable Weighted Random Sampling mode (default: False)")
    parser.add_argument("--cw", action="store_true", help="Enable Class Weighted mode (default: False)")
    parser.add_argument("--dataAug", action="store_true", help="Enable Data Augmentation mode (default: False)")
    parser.add_argument("--os", action="store_true", help="Enable OverSampling mode (default: False)")

    ## Model Configuration
    parser.add_argument("--imagemodel", type=int, choices=[0, 1, 2, 3], default=3,
                        help="Select RadDino type: 0=ResNet50, 1=MAIRA-1, 2=MAIRA-2, 3=WithoutImage (default: 3)")
    parser.add_argument("--raddinohead", action="store_true", help="Enable PreTrained Head mode (default: False)")
    parser.add_argument("--textmodel", type=int, choices=[0, 1, 2, 3, 4], default=4,
                        help="Select model type: 0=BertTokenizer, 1=BioBert, 2=CXR-Bert, 3=CXRBert-Especial, 4=WithoutText (default: 4)")
    parser.add_argument("--freezeImage", action="store_true", help="Enable Image Freeze RadDino Backbone mode (default: False)")
    parser.add_argument("--freezeText", action="store_true", help="Enable Text Freeze Backbone mode (default: False)")

    ## Aux modes
    parser.add_argument("--label_count", type=int, choices=[25, 20, 15, 10,5], default=25,
                        help="Select Label counts: 25 label, 20 label, 15 label, 10 label, 5 label, (default: 25)")
    args = parser.parse_args()

    main(args.wrs, args.cw,args.dataAug,args.os, args.imagemodel, args.raddinohead,args.textmodel,args.freezeImage,args.freezeText, args.label_count)



