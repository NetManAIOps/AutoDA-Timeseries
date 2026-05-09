from . import AutoDA_Timeseries


AVAILABLE_TSA={
    e.__name__.split('.')[-1]:e.Model for e in [AutoDA_Timeseries]
}

AVAILABLE_TSA["AutoDA-Timeseries"] = AutoDA_Timeseries.Model

def get_auto_augment_class(tsa:str):
    if tsa not in AVAILABLE_TSA:
        raise NotImplementedError(f"Unknown TSA:{tsa}(options:{list(AVAILABLE_TSA.keys())})")
    return AVAILABLE_TSA[tsa]
