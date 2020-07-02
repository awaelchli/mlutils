import os

import torch
from PIL import Image
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import ToTensor
from typing import List, Tuple, Dict, Union, Type, Optional

from mlutils.dataset import TransformationMixin
from mlutils.image_transforms import RandomCropper
from mlutils.optical_flow.io import read_flo


class Sequence(Dataset):

    def __init__(self, root_folder: str, name: str, subset: str = 'training'):
        super().__init__()
        self.root_folder = root_folder
        self.name = name
        self.subset = subset
        self.clean_files = self._collect_image_files('clean')
        self.final_files = self._collect_image_files('final')

    def _collect_image_files(self, _type: str) -> List[Path]:
        files = Path(os.path.join(self.root_folder, self.subset, _type, self.name)).glob(f'**/*.png')
        files = list(files)
        files.sort()
        return files

    def __len__(self) -> int:
        return len(self.clean_files) - 1

    def __getitem__(self, item: int) -> Dict[str, Union[Tuple[str, str], str]]:
        sample = {
            'clean': (str(self.clean_files[item]), str(self.clean_files[item + 1])),
            'final': (str(self.final_files[item]), str(self.final_files[item + 1])),
        }
        return sample


class TrainingSequence(Sequence):

    def __init__(self, root_folder: str, name: str):
        super().__init__(root_folder, name, subset='training')
        self.occlusion_files = self._collect_image_files('occlusions')
        self.invalid_files = self._collect_image_files('invalid')
        self.flow_files = self._collect_flow_files()

    def _collect_flow_files(self) -> List[Path]:
        files = Path(os.path.join(self.root_folder, self.subset, 'flow', self.name)).glob(f'**/*.flo')
        files = list(files)
        files.sort()
        return files

    def __getitem__(self, item: int) -> Dict[str, Union[Tuple[str, str], str]]:
        sample = super().__getitem__(item)
        sample['flow'] = str(self.flow_files[item])
        sample['occlusion'] = str(self.occlusion_files[item])
        sample['invalid'] = str(self.invalid_files[item])
        return sample


class TestSequence(Sequence):

    def __init__(self, root_folder: str, name: str):
        super().__init__(root_folder, name, subset='test')


class MPISintelIndex(Dataset):

    def __init__(self, root_folder: str, subset: str = 'training', sequence_names: Optional[List[str]] = None):
        super().__init__()
        self.root_folder = root_folder
        self.subset = subset
        sequence = self.__get_sequence_class()
        sequence_names = self.__get_sequene_names() if not sequence_names else sequence_names
        self.sequences = ConcatDataset(
            [sequence(root_folder, name) for name in sequence_names]
        )

    def __get_sequence_class(self) -> Type[Union[TestSequence, TrainingSequence]]:
        if self.subset == 'test':
            cls = TestSequence
        else:
            cls = TrainingSequence
        return cls

    def __get_sequene_names(self) -> List[str]:
        path = Path(self.root_folder, self.subset, 'clean')
        names = [x.name for x in path.iterdir() if x.is_dir()]
        names.sort()
        return names

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, item) -> Dict[str, Union[Tuple[str, str], str]]:
        return self.sequences[item]


class MPISintel(Dataset, TransformationMixin):

    def __init__(
            self,
            root_folder: str,
            subset: str,
            contents: Tuple[str, ...] = ('clean', 'flow'),
            sequences: Optional[List[str]] = None,
            random_crop: bool = False,
            crop_size: Tuple[int, int] = (128, 128),
    ):
        super().__init__()
        assert subset in ['training', 'test']
        assert set(contents).issubset({'clean', 'final', 'flow', 'occlusion', 'invalid', 'paths'})
        self.subset = subset
        self.contents = contents
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.index = MPISintelIndex(root_folder, subset, sequences)
        self.set_transformation(ToTensor(), 'image')
        self.set_transformation(ToTensor(), 'mask')
        self.set_transformation(None, 'flow')

    @property
    def is_training(self) -> bool:
        return self.subset == 'training'

    @property
    def is_test(self) -> bool:
        return self.subset == 'test'

    @property
    def frame_size(self) -> Tuple[int, int]:
        # height and width
        return 436, 1024

    def load_image(self, file: str) -> Tensor:
        img = Image.open(file)
        img = self.transform(img, groups='image')
        return img

    def load_mask(self, file: str) -> Tensor:
        img = Image.open(file)
        img = self.transform(img, 'mask')
        return img

    def load_flow(self, file: str) -> Tensor:
        flow = read_flo(file)
        flow = torch.as_tensor(flow).permute(2, 0, 1)
        flow = self.transform(flow, 'flow')
        return flow

    def __post_process_sample(self, sample: Dict[str, Union[Tensor, Tuple[Tensor, Tensor]]]) -> Dict[
        str, Union[Tensor, Tuple[Tensor, Tensor]]
    ]:
        if not self.random_crop:
            return sample

        cropper = RandomCropper(original_size=self.frame_size, crop_size=self.crop_size)
        for k, v in sample.items():
            if k == 'clean' or k == 'final':
                sample[k] = tuple(map(cropper, v))
            if k == 'flow':
                sample[k] = cropper(v)
            if k == 'occlusion' or k == 'invalid':
                sample[k] = cropper(v, mode='nearest')

        return sample

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> Dict[str, Union[Tensor, Tuple[Tensor, Tensor]]]:
        files = self.index[item]
        sample = dict()

        if 'paths' in self.contents:
            sample['paths'] = files
        if 'clean' in self.contents:
            sample['clean'] = tuple(map(self.load_image, files['clean']))
        if 'final' in self.contents:
            sample['final'] = tuple(map(self.load_image, files['final']))

        # test:
        if self.is_test:
            return self.__post_process_sample(sample)

        # training:
        if 'flow' in self.contents:
            sample['flow'] = self.load_flow(files['flow'])
        if 'occlusion' in self.contents:
            sample['occlusion'] = self.load_mask(files['occlusion'])        # TODO: check range
        if 'invalid' in self.contents:
            sample['invalid'] = self.load_mask(files['invalid'])            # TODO: check range
        return self.__post_process_sample(sample)


if __name__ == '__main__':
    data = MPISintel(
        '/home/adrian/Datasets/MPI-Sintel',
        subset='training',
        contents=('clean', 'final', 'flow', 'paths'),
        sequences=['market_2'],
        random_crop=False,
    )
    print(data[0]['clean'][0].shape)
    print(len(data[0]))
    print(data[0]['paths'])
