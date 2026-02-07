# Lazy imports: dnr/musdb datamodules pull in pedalboard (training-only).
# Inference only needs _types; avoid loading datamodules at package load time.
def __getattr__(name):
    if name == "DivideAndRemasterDataModule":
        from .dnr.datamodule import DivideAndRemasterDataModule
        return DivideAndRemasterDataModule
    if name == "MUSDB18DataModule":
        from .musdb.datamodule import MUSDB18DataModule
        return MUSDB18DataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")