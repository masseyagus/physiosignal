from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from physiosignal.info import Info

# Futura implementación
class RawSignal:

    def __init__(self, data, sfrq, info:Info, anotaciones, first_samp):
        pass

    def get_data(self, picks, start, stop, reject, times):
        pass

    def drop_channels(self, ch_names) -> RawSignal:
        pass

    def crop(self, tmin, tmax) -> RawSignal:
        pass

    def describe(self):
        pass

    def filter(self, low_freq, high_freq, notch_freq, order) -> RawSignal:
        pass

    def pick(self, picks) -> RawSignal:
        pass

    def set_anotaciones(self, anotaciones):
        pass

    def plot(self, picks, start, duration, show_anotaciones):
        pass  

    def __getitem__(self):
        pass