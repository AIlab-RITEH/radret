from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    def __init__(self, images, labels, irma_code, transform=None, albumentation=False):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.irma_code = irma_code
        self.albumentation = albumentation

    def __getitem__(self, index):
        sample = self.images[index]
        if self.transform:
            if (self.albumentation):
                sample = self.transform(image=sample)["image"]
            else:
                sample = self.transform(sample)
        return sample, self.labels[index], self.irma_code[index]

    def __len__(self):
        return len(self.images)
