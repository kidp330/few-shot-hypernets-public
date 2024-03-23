from pathlib import Path

save_dir                    = Path('./save/')
data_dir: dict[str, Path]   = {}
data_dir['CUB']             = Path('./filelists/CUB')
data_dir['miniImagenet']    = Path('./filelists/miniImagenet/')
data_dir['omniglot']        = Path('./filelists/omniglot/')
data_dir['emnist']          = Path('./filelists/emnist/')
kernel_type                 = 'bncossim' #'nn' #linear, rbf, spectral (regression only), matern, poli1, poli2, cossim, bncossim
