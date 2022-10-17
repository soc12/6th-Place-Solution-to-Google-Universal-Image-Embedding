import os
import time
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import cv2
import torch
import open_clip
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from utils import ArcMarginProductSubcenter, ArcFaceLossAdaptiveMargin
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.optim import Adam
import glob
from sklearn.decomposition import PCA
import pickle
from torchvision import transforms
from zipfile import ZipFile

# ====================================================

CFG = {
    "debug": False,
    "num_workers": 4,
    'num_classes': 43625,
    "model_name": 'clip',
    "size": 224,
    "epochs": 4,
    "batch_size": 32,
    "seed": 42,
    "loss": "ArcMarginProductSubcenter",
    'm': 0.45,
    's': 30,
    'm_low': 0.05
}

TRAIN_PATH = './data/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_dirs():
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('pca'):
        os.makedirs('pca')
    if not os.path.exists('submissions'):
        os.makedirs('submissions')
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(seed=CFG['seed'])
create_dirs()


class TrainDataset(Dataset):

    def __init__(self, filenames, labels, transform):
        self.filenames = filenames  # df['image_id'].values
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = self.filenames[idx]
        file_path = f'{TRAIN_PATH}{image}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (CFG['size'], CFG['size']))
        augmented = self.transform(image=image)
        image = augmented['image']
        label = self.labels[idx]
        return image, label


class ValidDataset(Dataset):

    def __init__(self, filenames, transform):
        self.filenames = filenames  # df['image_id'].values
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = self.filenames[idx]
        file_path = f'{TRAIN_PATH}{image}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (CFG['size'], CFG['size']))
        augmented = self.transform(image=image)
        image1 = augmented['image']
        return image1


data_transforms = {
    "train": A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_lower=99, quality_upper=100),
        A.RandomBrightnessContrast(0.1, 0.1, p=1),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, border_mode=0, p=0.7),
        A.RGBShift(5, 5, 5),
        A.Cutout(p=0.1),
        A.Normalize(
        ),
        ToTensorV2()], p=1.),
    "valid": A.Compose([
        A.Normalize(),
        ToTensorV2()], p=1.)
}


class ArcNet(nn.Module):
    def __init__(self, s, m):
        super(ArcNet, self).__init__()
        num_outputs = CFG["num_classes"]
        channel_size = 256
        self.model = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')[0].visual
        in_features = 1024
        self.model.head = nn.Sequential(nn.BatchNorm1d(in_features), nn.Dropout(0.2),
                                        nn.Linear(in_features, channel_size))
        self.arc = ArcMarginProductSubcenter(channel_size, num_outputs, 3)
        self.margin_fn = ArcFaceLossAdaptiveMargin(m, CFG['num_classes'], s)
        self.pool = nn.AdaptiveAvgPool1d(64)

    def forward(self, images, labels=None):
        features = self.model(images)
        features = self.model.head(features)
        if labels is not None:
            out = self.arc(features)
            return self.margin_fn(out, labels)
        features = self.pool(features)
        return features

    def logits(self, features, labels):
        return self.arc(features, labels)


def calculate_mAP(nbrs, freq):
    ap = 0
    for count, i in enumerate(nbrs):
        groundtruths = freq[1][np.where(freq[0] == valid_labels[count])]
        true_pos = np.where(valid_labels[i] == valid_labels[count])[0]
        ap += np.sum((np.arange(len(true_pos)) + 1) / (true_pos + 1)) / groundtruths
        cat_dict[cat_labels[count]] += np.sum((np.arange(len(true_pos)) + 1) / (true_pos + 1)) / groundtruths
    return ap / (count + 1)


if __name__ == '__main__':
    df = pd.read_csv("./data/train.csv")
    train_paths, train_labels = df[df['set'] == 'train'][['path', 'encoded_labels']].values.T
    valid_paths, valid_labels = df[df['set'] == 'valid'][['path', 'encoded_labels']].values.T
    cat_labels = df[df['set'] == 'valid']['cat_label'].values
    trainset = TrainDataset(train_paths, train_labels, data_transforms['train'])
    validset = ValidDataset(valid_paths, data_transforms['valid'])

    trainloader = DataLoader(trainset,
                             shuffle=True,
                             num_workers=4, pin_memory=True, batch_size=CFG['batch_size'])
    validloader = DataLoader(validset,
                             shuffle=False,
                             num_workers=4, pin_memory=True, batch_size=CFG['batch_size'])

    tmp = np.sqrt(1 / np.sqrt(df['encoded_labels'].value_counts().sort_index().values))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * CFG['m'] + CFG['m_low']
    model = ArcNet(CFG['s'], m=margins)
    model = model.to(device)

    def criterion(outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)


    final_block = 14
    blocks = ['resblocks.' + str(i) for i in range(31, 14, -1)]
    # criterion = nn.CrossEntropyLoss().to(device)
    # model.load_state_dict(
    #     torch.load('clip_H_14_arc_withnewdata_val[0.64657786]_epoch5_256emb_s30m0.5_31-5res_unfreeze15-4afterepoch5.pt')["model_state_dict"])
    for i in model.named_parameters():
        if "head" in i[0] or 'arc' in i[0] or any(substring in i[0] for substring in blocks):
            i[1].requires_grad = True
        else:
            i[1].requires_grad = False
        print(i[0], i[1].requires_grad)

    optimizer = Adam(
        [
            {"params": model.model.transformer.parameters(), "lr": 1e-7},
            {"params": model.model.head.parameters(), "lr": 1e-4},
            {"params": model.arc.parameters(), "lr": 1e-4},
        ],
        lr=1e-7,
    )

    freq = np.unique(valid_labels, return_counts=True)
    features = []
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, CFG['epochs'] + 1):
        cat_dict = dict.fromkeys(np.unique(cat_labels), 0)
        t0 = time.time()
        model.train()
        running_loss = 0.0
        acc = 0.0
        t0 = time.time()
        for i, element in enumerate(tqdm(trainloader)):
            imgs, targets = element[0].to(device), element[1].to(device)
            bsz = targets.shape[0]
            with torch.cuda.amp.autocast():
                pred = model.forward(imgs, targets)
                loss = criterion(pred, targets)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        print("Epoch {} Loss {}".format(epoch, running_loss / (i + 1)))

        model.eval()
        desc = []
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for i, element in enumerate(validloader):
                    pred = model.forward(element.cuda().float())
                    desc.append(pred)
        features = [item.cpu().numpy().reshape((64)) for sublist in desc for item in sublist]
        features = np.array(features)
        features = np.nan_to_num(features)
        nbrs = NearestNeighbors(n_neighbors=1000, metric='euclidean').fit(features)
        _, neighbors_trn = nbrs.kneighbors(features)
        ap = calculate_mAP(neighbors_trn, freq)
        print(" Validation mAP = ", ap)
        a = {k: v / sum(cat_labels == k) for k, v in cat_dict.items()}
        print(a)
        print(f'Time taken: {((time.time() - t0) / 60):.3f} mins')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f'./checkpoints/clip_vit-h14_val{ap}_epoch{epoch}.pt')

    model.pool = nn.Identity()
    # Fit PCA on 130k images dataset #
    imgs = glob.glob("./data/130k_kaggle/*/*")
    imgs = [i.split("/", 2)[-1] for i in imgs]
    pcaset = ValidDataset(imgs, data_transforms['valid'])
    pcaloader = DataLoader(pcaset,
                           shuffle=False,
                           num_workers=4, pin_memory=True, batch_size=64)
    model.eval()
    desc = []
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for i, element in enumerate(tqdm(pcaloader)):
                pred = model(element.cuda().float())
                desc.append(pred)
    features = [item.cpu().numpy().reshape((256)) for sublist in desc for item in sublist]
    features = np.array(features)
    features = np.nan_to_num(features)
    pca = PCA(n_components=64)
    pca.fit(features)
    with open('./pca/pca.pkl', 'wb') as f:
        pickle.dump(pca, f)


    class FinalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_clip = model

            self.pca_mean_clip = torch.nn.Parameter(torch.tensor(pca.mean_))
            self.pca_matrix_clip = torch.nn.Parameter(torch.tensor(pca.components_))

        def preprocess_image_clip(self, x):
            x = transforms.functional.resize(x, size=[224, 224])
            x = x / 255.0
            x = transforms.functional.normalize(x,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
            return x

        def forward(self, x):
            x_clip = self.preprocess_image_clip(x)
            x_clip = self.encoder_clip(x_clip)

            x_clip = torch.nn.functional.normalize(x_clip)
            x_clip = (x_clip - self.pca_mean_clip) @ (self.pca_matrix_clip).T

            return x_clip


    model = FinalModel().to(device).eval()
    saved_model = torch.jit.trace(model.eval(), torch.randn((1, 3, 224, 224)).to(device))
    saved_model.save('./models/final_model.pt')

    with ZipFile('./submissions/submission.zip', 'w') as zip:
        zip.write('./models/final_model.pt', arcname='final_model.pt')
    # model = torch.jit.load('./models/final_model.pt').to('cuda').eval()
    #
    # input_batch = torch.rand(1, 3, 224, 224).to('cuda')
    # with torch.no_grad():
    #     embedding = model(input_batch).cpu().data.numpy()
    # print(embedding.shape)
