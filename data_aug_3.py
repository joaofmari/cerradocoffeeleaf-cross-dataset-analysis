# Author: João Fernando Mari
# joaofmari.github.io
# https://github.com/joaofmari

from torchvision import transforms

def get_da(da, DS_MEAN=[0.485, 0.456, 0.406], DS_STD=[0.229, 0.224, 0.225]):
    """
    """
    if da == 1:  
        # Resize(224)
        data_transforms = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(DS_MEAN, DS_STD)
        ])

    elif da == 2: 
        # Resise(256) + CenterCrop(224)
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(DS_MEAN, DS_STD)
        ])

    elif da == 3: # **** HP optimizatio AND Without DA  ****
        # RandomResizedCrop (224). 
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(DS_MEAN, DS_STD)
        ])
        
    # **** DATA AUGMENTATION (1) ****
    elif da == 11: 
        # Data augmentation base
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(-90, 90)),
            transforms.Resize(size=(224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.3)),
            transforms.Normalize(DS_MEAN, DS_STD),
        ])

    elif da == 12: # **** With DA ****
        # Data augmentation base. NO HUE. 
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(-90, 90)),
            transforms.Resize(size=(224, 224)),
            ### transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.3)),
            transforms.Normalize(DS_MEAN, DS_STD),
        ])

    elif da == 13: 
        # Data augmentation base. NO HUE, NO RandomErasing.
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize(size=(224, 224)),
            ### transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            ### transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            transforms.Normalize(DS_MEAN, DS_STD),
        ])
        
    elif da == 14: 
        # Data augmentation base. NO Jitter, NO RandomErasing.
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize(size=(224, 224)),
            ###transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            ###transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)),
            transforms.Normalize(DS_MEAN, DS_STD),
        ])

    # **** DATA AUGMENTATION (2) ****
    elif da == 21: 
        # Data augmentation base
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)),
            transforms.Normalize(DS_MEAN, DS_STD),
        ])

    elif da == 22: # **** With DA ****
        # Data augmentation base. NO HUE. 
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            ### transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)),
            transforms.Normalize(DS_MEAN, DS_STD),
        ])

    elif da == 23: 
        # Data augmentation base. NO HUE, NO RandomErasing.
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            ### transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            ### transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            transforms.Normalize(DS_MEAN, DS_STD),
        ])
        
    elif da == 24: 
        # Data augmentation base. NO Jitter, NO RandomErasing.
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            ###transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            ###transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)),
            transforms.Normalize(DS_MEAN, DS_STD),
        ])

    # **** DATA AUGMENTATION (3) ****
    elif da == 31: 
        # Data augmentation base
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)),
            transforms.Normalize(DS_MEAN, DS_STD),
        ])

    elif da == 32: # **** With DA ****
        # Data augmentation base. NO HUE. 
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            ### transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)),
            transforms.Normalize(DS_MEAN, DS_STD),
        ])

    elif da == 33: 
        # Data augmentation base. NO HUE, NO RandomErasing.
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            ### transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            ### transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            transforms.Normalize(DS_MEAN, DS_STD),
        ])
        
    elif da == 34: 
        # Data augmentation base. NO Jitter, NO RandomErasing.
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            ###transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            ###transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)),
            transforms.Normalize(DS_MEAN, DS_STD),
        ])
 
    return data_transforms

