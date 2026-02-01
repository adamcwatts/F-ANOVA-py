from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("f-anova-py")
except PackageNotFoundError:
    pass


from functionalANOVA.core.fanova import (
    functionalANOVA,
    ANOVALabels,
    ANOVAUnits,
    ANOVATables,
    ANOVAMethods,
    ANOVAGroups,
)

__all__ = [
    "functionalANOVA",
    "ANOVALabels",
    "ANOVAUnits",
    "ANOVATables",
    "ANOVAMethods",
    "ANOVAGroups",
]
