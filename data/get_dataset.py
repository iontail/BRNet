from src.datasets.dataset_256 import ImageFolderMaskDataset_256
from src.datasets.dataset_256_val import ImageFolderMaskDataset_256_VAL
from src.datasets.dataset_512 import ImageFolderMaskDataset_512
from src.datasets.dataset_512_val import ImageFolderMaskDataset_512_VAL


from src.datasets.coreap_datacollator import CoReaPDataCollator

def get_dataset(args):

    if args.is_train:
        if args.data_name == '256':
            train_dataset = ImageFolderMaskDataset_256(path = './data/celeba_hq/train')
            valid_dataset = ImageFolderMaskDataset_256_VAL(path = './data/celeba_hq/val') 
        elif args.data_name == '512':
            train_dataset = ImageFolderMaskDataset_512(path = './data/celeba_hq/train')
            valid_dataset = ImageFolderMaskDataset_512_VAL(path = './data/celeba_hq/val')
        data_collator = CoReaPDataCollator(args)

        return train_dataset, valid_dataset, data_collator

    else:
        test_data = None
        test_dataset = CoReaPDataset(args, test_data)
        data_collator = CoReaPDataCollator(args)

        return test_dataset, None, data_collator


