from typing import Union, List, Tuple, Callable, Optional

from torchvision.transforms import Compose, ToTensor, ToPILImage


class TransformationMixin(object):

    def __init__(self):
        super().__init__()
        self.__transforms = dict(
            default=Compose([]),  # identity
            to_tensor=ToTensor(),
            to_pil_image=ToPILImage(),
        )

    @property
    def transformation_groups(self):
        return self.__transforms.keys()

    def set_transformation(self, transform: Optional[Callable], group: str = 'default'):
        """ If transformation is None, defaults to Identity transformation. """
        self.__transforms[group] = Compose([])
        if transform:
            self.append_transformation(transform, group)

    def append_transformation(self, transform: Callable, group: str = 'default'):
        self.__transforms[group].transforms.append(transform)

    def transform(self, item, groups: Union[str, List[str], Tuple[str, ...]] = 'default'):
        if isinstance(groups, str):
            return self.__transforms[groups](item)
        if isinstance(groups, (list, tuple)):
            for group in groups:
                item = self.__transforms[group](item)
            return item
