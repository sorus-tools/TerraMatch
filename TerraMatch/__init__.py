def classFactory(iface):
    from .terramatch import TerraMatchPlugin

    return TerraMatchPlugin(iface)
