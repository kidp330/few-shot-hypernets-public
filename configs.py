from pathlib import Path

from parsers.types.general import Dataset


save_dir: Path = Path('./save/')
data_dir: dict[Dataset, Path] = {
    'CUB': Path('./filelists/CUB/'),
    'miniImagenet': Path('./filelists/miniImagenet/'),
    'omniglot': Path('./filelists/omniglot/'),
    'emnist': Path('./filelists/emnist/'),
}
# 'nn' #linear, rbf, spectral (regression only), matern, poli1, poli2, cossim, bncossim
kernel_type = 'bncossim'
