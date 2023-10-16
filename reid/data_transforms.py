from torchvision import transforms
from data_augment import LGT, Fuse_RGB_Gray_Sketch, Fuse_Gray


def get_train_transforms(dataset, ratio=1, transformer_model=False):
    if dataset in ("market1501", "dukemtmc"):
        if transformer_model:
            transform_train = transforms.Compose([
                transforms.Resize((448, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.Pad(10),
                transforms.RandomCrop((448, 224)),
                LGT(),
                # transforms.ToTensor(),
                transforms.RandomErasing(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize((256, int(256 * ratio))),  # interpolation=3
                transforms.RandomHorizontalFlip(),
                transforms.Pad(10),
                transforms.RandomCrop((256, int(256 * ratio))),
                Fuse_Gray(0.35, 0.05),
                # transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                transforms.RandomErasing(),
            ])
    else:
        if transformer_model:
            transform_train = transforms.Compose([
                transforms.Resize((224, 224)),  # interpolation=3
                transforms.RandomHorizontalFlip(),
                transforms.Pad(10),
                transforms.RandomCrop((224, 224)),
                Fuse_Gray(0.35, 0.05),
                # transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                transforms.RandomErasing(),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize((224, int(ratio * 224))),  # interpolation=3
                transforms.RandomHorizontalFlip(),
                transforms.Pad(10),
                transforms.RandomCrop((224, int(ratio * 224))),
                Fuse_Gray(0.35, 0.05),
                # transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                transforms.RandomErasing(),
            ])
    return transform_train


def get_inference_transforms(dataset, ratio=1, transformer_model=False, strong_inference=False):
    if dataset in ("market1501", "dukemtmc"):
        if strong_inference:
            if transformer_model:
                transform_test = transforms.Compose([transforms.Resize((448, 224)),
                                                     transforms.Pad(10),
                                                     transforms.RandomCrop((448, 224)),  # experimental
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                          std=(0.229, 0.224, 0.225)),
                                                     ]
                                                    )
            else:
                transform_test = transforms.Compose([transforms.Resize((256, int(256 * ratio))),
                                                     transforms.Pad(10),
                                                     transforms.RandomCrop((256, int(256 * ratio))),  # experimental
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                          std=(0.229, 0.224, 0.225)),
                                                     ]
                                                    )
        else:
            if transformer_model:
                transform_test = transforms.Compose([transforms.Resize((448, 224)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                          std=(0.229, 0.224, 0.225)),
                                                     ]
                                                    )
            else:
                transform_test = transforms.Compose([transforms.Resize((256, int(256 * ratio))),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                          std=(0.229, 0.224, 0.225)),
                                                     ]
                                                    )
    else:
        if strong_inference:
            if transformer_model:
                transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                                     transforms.Pad(10),
                                                     transforms.RandomCrop((224, 224)),  # experimental
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                          std=(0.229, 0.224, 0.225)),
                                                     ]
                                                    )
            else:
                transform_test = transforms.Compose([transforms.Resize((224, int(ratio * 224))),
                                                     transforms.Pad(10),
                                                     transforms.RandomCrop((224, int(ratio * 224))),  # experimental
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                          std=(0.229, 0.224, 0.225)),
                                                     ]
                                                    )
        else:
            if transformer_model:
                transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                          std=(0.229, 0.224, 0.225)),
                                                     ]
                                                    )
            else:
                transform_test = transforms.Compose([transforms.Resize((224, int(ratio * 224))),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                          std=(0.229, 0.224, 0.225)),
                                                     ]
                                                    )
    return transform_test


def get_inference_transforms_flipped(dataset, ratio=1, transformer_model=False, strong_inference=False):
    if dataset in ("market1501", "dukemtmc"):
        if strong_inference:
            if transformer_model:
                transform_test = transforms.Compose([transforms.Resize((448, 224)),
                                                     transforms.RandomHorizontalFlip(p=1.0),
                                                     transforms.Pad(10),
                                                     transforms.RandomCrop((448, 224)),  # experimental
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                          std=(0.229, 0.224, 0.225)),
                                                     ]
                                                    )
            else:
                transform_test = transforms.Compose([transforms.Resize((256, int(256 * ratio))),
                                                     transforms.RandomHorizontalFlip(p=1.0),
                                                     transforms.Pad(10),
                                                     transforms.RandomCrop((256, int(256 * ratio))),  # experimental
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                          std=(0.229, 0.224, 0.225)),
                                                     ]
                                                    )
        else:
            if transformer_model:
                transform_test = transforms.Compose([transforms.Resize((448, 224)),
                                                     transforms.RandomHorizontalFlip(p=1.0),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                          std=(0.229, 0.224, 0.225)),
                                                     ]
                                                    )
            else:
                transform_test = transforms.Compose([transforms.Resize((256, int(256 * ratio))),
                                                     transforms.RandomHorizontalFlip(p=1.0),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                          std=(0.229, 0.224, 0.225)),
                                                     ]
                                                    )
    else:
        if strong_inference:
            if transformer_model:
                transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                                     transforms.RandomHorizontalFlip(p=1.0),
                                                     transforms.Pad(10),
                                                     transforms.RandomCrop((224, 224)),  # experimental
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                          std=(0.229, 0.224, 0.225)),
                                                     ]
                                                    )
            else:
                transform_test = transforms.Compose([transforms.Resize((224, int(ratio * 224))),
                                                     transforms.RandomHorizontalFlip(p=1.0),
                                                     transforms.Pad(10),
                                                     transforms.RandomCrop((224, int(ratio * 224))),  # experimental
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                          std=(0.229, 0.224, 0.225)),
                                                     ]
                                                    )
        else:
            if transformer_model:
                transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                                     transforms.RandomHorizontalFlip(p=1.0),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                          std=(0.229, 0.224, 0.225)),
                                                     ]
                                                    )
            else:
                transform_test = transforms.Compose([transforms.Resize((224, int(ratio * 224))),
                                                     transforms.RandomHorizontalFlip(p=1.0),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                          std=(0.229, 0.224, 0.225)),
                                                     ]
                                                    )
    return transform_test
