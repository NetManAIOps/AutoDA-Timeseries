from abc import abstractmethod, ABC
from autoaugment import get_auto_augment_class, AutoAugmentBasic
from utils import GlobalConfig




class ExpBasic(ABC):
    def __init__(self, config:GlobalConfig):
        self.config = config
        self.loaded_data = dict()
        self.model_class = get_auto_augment_class(config.args.tsa)
        self.device = config.device
        self.model = self._build_model()

    @abstractmethod
    def _build_model(self) -> AutoAugmentBasic:
        raise NotImplementedError()

    @abstractmethod
    def _get_data(self, flag:str, load_as:str):
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def vali(self, vali_data, vali_loader, criterion):
        raise NotImplementedError()

    @abstractmethod
    def test(self, load_checkpoint:bool=False):
        raise NotImplementedError()