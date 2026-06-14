import numpy as np
import torch
from aeon.datasets import load_classification

from utils import GlobalConfig
from utils.socks_proxy import ProxyContext
from .dataset_basic import BasicDataset
from .tsfeature_extractors import get_feature_extractor, FeatureExtractor


class DefaultClassificationDataset(BasicDataset):
    def __init__(self, config:GlobalConfig, flag):
        super().__init__(config, flag)

    def _load_data(self, root_path, dataset, flag):
        assert flag in {"TRAIN","TEST"}
        print(f"[{self.__class__.__name__}] loading data (flag={flag})")
        # (batch, n_channel, seq_len) and (batch, n_channel, 1)
        with ProxyContext(self.config):
            self.X, self.Y, metadata = load_classification(dataset, split=flag, extract_path=root_path,
                                                           return_metadata=True)
            self.X = np.array(self.X)

        print(f"[{self.__class__.__name__}] data loaded (flag={flag}) input size={self.X.shape}")
        self.n_classes = len(metadata["class_values"])
        class_map = {v:i for i,v in enumerate(metadata["class_values"])}
        self.Y = np.array([class_map[e] for e in self.Y], dtype=np.int8)

        # self.fe:FeatureExtractor = get_feature_extractor(self.config.args.feature_extractor)(self.config, None)
        # # (batch, n_channel, cond_dim)
        # self.F = self.fe(self.X)
        # print(f"self.F: {self.F.shape}")
        # exit(0)

        self.fe: FeatureExtractor = get_feature_extractor(self.config.args.feature_extractor)(self.config, None)
        # (batch, n_channel, cond_dim)
        self.F = self.fe(self.X)

        # ===========================
        # Remove specific features
        # ===========================

        FEATURE_NAMES = [
            "DN_HistogramMode_5", 
            "DN_HistogramMode_10", 
            "CO_f1ecac",
            "CO_FirstMin_ac",
            "CO_HistogramAMI_even_2_5",
            "CO_trev_1_num",
            "MD_hrv_classic_pnn40",
            "SB_BinaryStats_mean_longstretch1",
            "SB_TransitionMatrix_3ac_sumdiagcov",
            "PD_PeriodicityWang_th0_01",
            "CO_Embed2_Dist_tau_d_expfit_meandiff",
            "IN_AutoMutualInfoStats_40_gaussian_fmmi",
            "FC_LocalSimple_mean1_tauresrat",
            "DN_OutlierInclude_p_001_mdrmd",
            "DN_OutlierInclude_n_001_mdrmd",
            "SP_Summaries_welch_rect_area_5_1",
            "SB_BinaryStats_diff_longstretch0",
            "SB_MotifThree_quantile_hh",
            "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
            "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
            "SP_Summaries_welch_rect_centroid",
            "FC_LocalSimple_mean3_stderr"
        ]

        # Remove all features whose names start with CO_.
        # indices = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith("MD_")]
        
        # indices = [i for i, name in enumerate(FEATURE_NAMES)
        #    if name.startswith(("MD_", "SB_"))]
        # if len(indices) > 0:
        #     self.F[:, :, indices] = 0.0

        self.X = torch.nan_to_num(torch.from_numpy(self.X), nan=0)
        self.Y = torch.nan_to_num(torch.from_numpy(self.Y), nan=0)
        self.F = torch.nan_to_num(torch.from_numpy(self.F), nan=0)
        self.mask = torch.ones((self.X.shape[0], self.X.shape[-1]), dtype=torch.bool, device=self.X.device)
        self._parse_dimensions()
        return self.X, self.Y, self.F, self.mask


AVAILABLE_CLASSIFICATION_DATASETS = {
    "Default": DefaultClassificationDataset
}
