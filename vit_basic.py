import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import copy

from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- LOADING THE DATA ---------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# We set the mean and standard deviation, respectively
cifar10_mean = torch.tensor([0.49139968, 0.48215827, 0.44653124])
cifar10_std = torch.tensor([0.24703233, 0.24348505, 0.26158767])

# We set the path's root of the dataset
data_path = 'C:\Datasets\CIFAR 10\data\cifar10'

# We set the batch size we are working on
batch_size = 128

# We set the training, validation and test dataset sizes, respectively
train_size = 1280
val_size = 640
test_size = 1280


class Cifar10_Dataset(Dataset):
    def __init__(self, train):
        self.transform = transforms.Compose([
            transforms.Resize(40),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std)
        ])

        self.dataset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=train,
            download=True
        )

        # We consider only a subset of CIFAR10 dataset (for time and computing resources)
        mask = list(range(0, len(self.dataset), 8))
        self.dataset = torch.utils.data.Subset(self.dataset, mask)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img, label


# We download the CIFAR10 dataset for training process, if it does not exist at the specified path
train_set = Cifar10_Dataset(True)

# We take the validation set as a part of the training set using the sized set above
train_set, val_set, _ = torch.utils.data.random_split(train_set,
                                                      [train_size, val_size, len(train_set) - train_size - val_size])

# We download the CIFAR10 dataset for testing, if it does not exist at the specified path
test_set = Cifar10_Dataset(False)

# We consider only a part of the data for testing (for computational cost reasons)
test_set, _ = torch.utils.data.random_split(test_set,
                                            [test_size, len(test_set) - test_size])

# We set the DataLoaders corresponding to each set
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)

val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)

test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- THE ARCHITECTURE ---------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# First, we define the Multi-Self Attention block
class MSA(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.K_embed = nn.Linear(input_dim, embed_dim, bias=False)
        self.Q_embed = nn.Linear(input_dim, embed_dim, bias=False)
        self.V_embed = nn.Linear(input_dim, embed_dim, bias=False)

        self.out_embed = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        batch_size, max_length, given_input_dim = x.shape

        assert given_input_dim == self.input_dim, "given_input_dim NOT EQUAL TO input_dim"
        assert max_length % self.num_heads == 0, "max_length NOT DIVISIBLE WITH num_heads"

        x = x.reshape(batch_size * max_length, -1)

        K = self.K_embed(x).reshape(batch_size, max_length, self.embed_dim)
        Q = self.Q_embed(x).reshape(batch_size, max_length, self.embed_dim)
        V = self.V_embed(x).reshape(batch_size, max_length, self.embed_dim)

        indiv_dim = self.embed_dim // self.num_heads

        K = K.reshape(batch_size, max_length, self.num_heads, indiv_dim)
        Q = Q.reshape(batch_size, max_length, self.num_heads, indiv_dim)
        V = V.reshape(batch_size, max_length, self.num_heads, indiv_dim)

        K = K.permute(0, 2, 1, 3)
        Q = Q.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        K = K.reshape(batch_size * self.num_heads, max_length, indiv_dim)
        Q = Q.reshape(batch_size * self.num_heads, max_length, indiv_dim)
        V = V.reshape(batch_size * self.num_heads, max_length, indiv_dim)

        K_T = K.permute(0, 2, 1)
        QK = torch.bmm(Q, K_T)

        d = self.embed_dim
        weights = torch.div(QK, np.sqrt(d))
        weights = F.softmax(weights, 2)

        w_V = torch.bmm(weights, V)
        w_V = w_V.reshape(batch_size, self.num_heads, max_length, indiv_dim)
        w_V = w_V.permute(0, 2, 1, 3)
        w_V = w_V.reshape(batch_size, max_length, self.embed_dim)

        out = self.out_embed(w_V)

        return out


# Second, we define an ViT layer
class ViTLayer(nn.Module):
    def __init__(self, num_heads, input_dim, embed_dim, mlp_hidden_dim, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim

        self.msa = MSA(input_dim, embed_dim, num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.w_o_dropout = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.layernorm1(x)
        x = self.msa(x)
        x = self.w_o_dropout(x)
        x = self.layernorm2(x)
        x = self.mlp(x)
        x = self.w_o_dropout(x)

        return x


# Third, we define the ViT architecture
class ViT(nn.Module):
    def __init__(self, patch_dim, image_dim, num_layers, num_heads, embed_dim, mlp_hidden_dim, num_classes, dropout):
        super().__init__()

        self.num_layers = num_layers
        self.patch_dim = patch_dim
        self.image_dim = image_dim
        self.input_dim = self.patch_dim * self.patch_dim * 3
        self.num_heads = num_heads

        self.patch_embedding = nn.Linear(self.input_dim, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, (image_dim // patch_dim) ** 2 + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embedding_dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([])

        for i in range(num_layers):
            self.encoder_layers.append(ViTLayer(num_heads, embed_dim, embed_dim, mlp_hidden_dim, dropout))

        self.mlp_head = nn.Linear(embed_dim, num_classes)
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, images):
        h = w = self.image_dim // self.patch_dim
        N = images.size(0)

        images = images.reshape(N, 3, h, self.patch_dim, w, self.patch_dim)
        images = torch.einsum("nchpwq -> nhwpqc", images)
        patches = images.reshape(N, h * w, self.input_dim)

        patch_embeddings = self.patch_embedding(patches)
        patch_embeddings = torch.cat([torch.tile(self.cls_token, (N, 1, 1)), patch_embeddings], dim=1)

        out = patch_embeddings + torch.tile(self.position_embedding, (N, 1, 1))
        out = self.embedding_dropout(out)

        add_len = (self.num_heads - out.shape[1]) % self.num_heads

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        out = torch.cat([out, torch.zeros(N, add_len, out.shape[2], device=device)], dim=1)

        for i in range(self.num_layers):
            out = self.encoder_layers[i](out)

        cls_head = self.layernorm(torch.squeeze(out[:, 0], dim=1))
        logits = self.mlp_head(cls_head)

        return logits


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- DEFINE SOME VITs ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def get_vit_tiny(num_classes=10, patch_dim=4, image_dim=32):
    return ViT(
        patch_dim=patch_dim,
        image_dim=image_dim,
        num_layers=12,
        num_heads=3,
        embed_dim=192,
        mlp_hidden_dim=768,
        num_classes=num_classes,
        dropout=0.1)


def get_vit_small(num_classes=10, patch_dim=4, image_dim=32):
    return ViT(
        patch_dim=patch_dim,
        image_dim=image_dim,
        num_layers=12,
        num_heads=6,
        embed_dim=384,
        mlp_hidden_dim=1536,
        num_classes=num_classes,
        dropout=0.1)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- TRAINING -----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def compute_accuracy(model, data_loader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Set the model to GPU if available
    model.eval()  # Set the model to the evaluation mode

    data_loss = 0.0
    data_acc = 0.0
    data_correct = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Set the data to GPU if available

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            data_loss += loss.item()
            data_correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

    data_loss = data_loss / (len(data_loader) * batch_size)
    data_acc = data_correct / (len(data_loader) * batch_size)

    print(f"\nThere are {data_correct} correct predictions --- out of {len(data_loader) * batch_size} elements")

    return data_loss, data_acc


def training(model, train_loader, val_loader, test_loader, num_epochs, criterion, optimizer):
    print("\n\n\n ----- Training Process of ViT -----")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Set the model to GPU if available

    best_accuracy = -np.inf
    best_weights = None
    best_epoch = 0

    for epoch in range(num_epochs):
        start_time = time.perf_counter()
        model.train()  # Set the model to the training mode

        train_loss = 0.0
        train_acc = 0.0
        train_total = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Set the data to GPU if available

            # Forward and Backward passes
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss
            train_acc += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

        train_loss = train_loss / (len(train_loader) * batch_size)
        train_acc = train_acc / (len(train_loader) * batch_size)

        # Take the model('s weights) with the best accuracy
        print(f"\n Computing the Validation Accuracy for Epoch {epoch + 1}:")
        validation_loss, validation_accuracy = compute_accuracy(model=model, data_loader=val_loader,
                                                                criterion=criterion)

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        print(f'\nEpoch: {epoch + 1: 2d} ===> Train Loss: {train_loss: .4f} --- Train Accuracy: {train_acc: .4f} ===> '
              f'Validation Loss: {validation_loss: .4f} --- Validation Accuracy: {validation_accuracy: .4f} ===>'
              f'Best Accuracy: {best_accuracy: .4f} ant epoch {best_epoch}')

    # Set the model('s weights) with the best accuracy
    model.load_state_dict(best_weights)

    print(f"\nComputing the Test Accuracy for the best model")
    test_loss, test_accuracy = compute_accuracy(model=model, data_loader=test_loader, criterion=criterion)
    print(f"The Test Loss is {test_loss: .4f} and the Test Accuracy is: {test_accuracy: .4f}")

    # We save the model to the specified path
    model_path = "../vit_basic.pth"
    torch.save(model, model_path)


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- MAIN ------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vit = get_vit_small().to(device)
    vit = torch.nn.DataParallel(vit)

    num_epochs = 2
    learning_rate = 1e-3
    weight_decay = 0.1

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        vit.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay
    )

    training(model=vit, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
             num_epochs=num_epochs, criterion=criterion, optimizer=optimizer)
