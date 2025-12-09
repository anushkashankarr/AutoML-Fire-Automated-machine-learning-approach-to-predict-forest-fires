import pkgutil
import importlib

__all__ = []

# Iterate through utils modules
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f"{__name__}.{module_name}")

    # 1️⃣ Expose module itself as utils.module_name
    globals()[module_name] = module
    __all__.append(module_name)

    # 2️⃣ Try auto-export symbols (functions/classes/constants)
    #     Case A: If module defines __all__, respect it
    if hasattr(module, "__all__"):
        for symbol in module.__all__:
            globals()[symbol] = getattr(module, symbol)
            __all__.append(symbol)

    # 3️⃣ Case B: If module has NO __all__, auto-export functions/classes
    else:
        for symbol_name in dir(module):
            if symbol_name.startswith("_"):
                continue  # skip private names

            symbol = getattr(module, symbol_name)

            # Export only functions & classes
            if callable(symbol) or isinstance(symbol, type):
                globals()[symbol_name] = symbol
                __all__.append(symbol_name)
