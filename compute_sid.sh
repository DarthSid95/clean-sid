set -e
# if [ "${CONDA_DEFAULT_ENV}" != "TF25" ]; then
# 	echo 'You are not in the <TF25> environment. Attempting to activate the RumiGAN environment. Please run "conda activate TF25" and try again if this fails.'
# 	condaActivatePath=$(which activate)
# 	source ${condaActivatePath} TF25
# fi

if [ "${CONDA_DEFAULT_ENV}" != "TF24" ]; then
	echo 'You are not in the <TF24> environment. Attempting to activate the TF24 environment. Please run "conda activate TF24" and try again if this fails.'
	condaActivatePath=$(which activate)
	source ${condaActivatePath} TF24
fi

###------ ###------ ###------ ###------ ###------ 
###------ SpiderGANs Rebuttal_Metrics ----- ###------ 
###------ ###------ ###------ ###------ ###------ 

###------ ###------ MNIST _Base
# python ./sid.py --model 'SpiderGANInterpol' --data 'MNIST' --noise 'NonPara' --reals_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_11052022/11052022_Base_non_para_mnist_dcgan_WGAN_R1_01/FID_reals_train' --fakes_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_11052022/11052022_Base_non_para_mnist_dcgan_WGAN_R1_01/FID_interpol_model_metrics' --order 1 --GPU '3' --SID_flag 0 --FID_flag 1 --KID_flag 1 --metric_mode 'clean'

# python ./sid.py --model 'SpiderGANInterpol' --data 'MNIST' --noise 'Gamma' --reals_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_11052022/11052022_Base_non_para_mnist_dcgan_WGAN_R1_01/FID_reals_train' --fakes_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_11052022/11052022_Base_gamma_mnist_dcgan_WGAN_R1_01/FID_interpol_model_metrics' --order 1 --GPU '3' --SID_flag 0 --FID_flag 1 --KID_flag 1 --metric_mode 'clean'

# python ./sid.py --model 'SpiderGANInterpol' --data 'MNIST' --noise 'Gaussian' --reals_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_11052022/11052022_Base_non_para_mnist_dcgan_WGAN_R1_01/FID_reals_train' --fakes_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_11052022/11052022_Base_gaussian_mnist_dcgan_WGAN_R1_01/FID_interpol_model_metrics' --order 1 --GPU '3' --SID_flag 0 --FID_flag 1 --KID_flag 1 --metric_mode 'clean'


###------ ###------ MNIST _Spider
# python ./sid.py --model 'SpiderGANInterpol' --data 'MNIST' --noise 'FMNIST' --reals_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_10052022/10052022_SpiderGAN_gaussian025_fmnist_mnist_deepconv_WGAN_R1_01/FID_reals_train' --fakes_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_10052022/10052022_SpiderGAN_gaussian025_fmnist_mnist_deepconv_WGAN_R1_01/FID_interpol2_model_metrics' --order 1 --GPU '' --SID_flag 0 --FID_flag 1 --KID_flag 1 --metric_mode 'clean' --batch_size 128

python ./sid.py --model 'SpiderGANInterpol' --data 'MNIST' --noise 'CIFAR10' --reals_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_10052022/10052022_SpiderGAN_gaussian025_fmnist_mnist_deepconv_WGAN_R1_01/FID_reals_train' --fakes_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_10052022/10052022_SpiderGAN_gaussian025_cifar10_mnist_deepconv_WGAN_R1_01/FID_interpol2_model_metrics' --order 1 --GPU '' --SID_flag 0 --FID_flag 1 --KID_flag 1 --metric_mode 'clean' --batch_size 128

python ./sid.py --model 'SpiderGANInterpol' --data 'MNIST' --noise 'TinyImgNet' --reals_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_10052022/10052022_SpiderGAN_gaussian025_fmnist_mnist_deepconv_WGAN_R1_01/FID_reals_train' --fakes_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_10052022/10052022_SpiderGAN_gaussian025_tinyimgnet_mnist_deepconv_WGAN_R1_01/FID_interpol2_model_metrics' --order 1 --GPU '' --SID_flag 0 --FID_flag 1 --KID_flag 1 --metric_mode 'clean' --batch_size 128

python ./sid.py --model 'SpiderGANInterpol' --data 'MNIST' --noise 'SVHN' --reals_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_10052022/10052022_SpiderGAN_gaussian025_fmnist_mnist_deepconv_WGAN_R1_01/FID_reals_train' --fakes_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_10052022/10052022_SpiderGAN_gaussian025_svhn_mnist_deepconv_WGAN_R1_01/FID_interpol2_model_metrics' --order 1 --GPU '' --SID_flag 0 --FID_flag 1 --KID_flag 1 --metric_mode 'clean' --batch_size 128

python ./sid.py --model 'SpiderGANInterpol' --data 'MNIST' --noise 'CelebA' --reals_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_10052022/10052022_SpiderGAN_gaussian025_fmnist_mnist_deepconv_WGAN_R1_01/FID_reals_train' --fakes_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_11052022/11052022_SpiderGAN_gaussian025_celeba_mnist_deepconv_WGAN_R1_01/FID_interpol2_model_metrics' --order 1 --GPU '' --SID_flag 0 --FID_flag 1 --KID_flag 1 --metric_mode 'clean' --batch_size 128

python ./sid.py --model 'SpiderGANInterpol' --data 'MNIST' --noise 'UkiyoE' --reals_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_10052022/10052022_SpiderGAN_gaussian025_fmnist_mnist_deepconv_WGAN_R1_01/FID_reals_train' --fakes_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/MNISTv4/Log_Folder_11052022/11052022_SpiderGAN_gaussian025_ukiyoe_mnist_deepconv_WGAN_R1_01/FID_interpol2_model_metrics' --order 1 --GPU '' --SID_flag 0 --FID_flag 1 --KID_flag 1 --metric_mode 'clean' --batch_size 128


# ###------ ###------ C10 _Base
# python ./sid.py --model 'SpiderGANInterpol' --data 'C10' --noise 'NonPara' --reals_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/CIFAR10/15052022_Base_non_para_cifar10_dcgan_WGAN_R1_01/FID_reals_train' --fakes_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/CIFAR10/15052022_Base_non_para_cifar10_dcgan_WGAN_R1_01/FID_interpol_model_metrics' --order 1 --GPU '' --SID_flag 0 --FID_flag 1 --KID_flag 1 --metric_mode 'clean'

# python ./sid.py --model 'SpiderGANInterpol' --data 'C10' --noise 'Gamma' --reals_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/CIFAR10/15052022_Base_non_para_cifar10_dcgan_WGAN_R1_01/FID_reals_train' --fakes_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/CIFAR10/15052022_Base_gamma_cifar10_dcgan_WGAN_R1_01/FID_interpol_model_metrics' --order 1 --GPU '3' --SID_flag 0 --FID_flag 1 --KID_flag 1 --metric_mode 'clean'

# python ./sid.py --model 'SpiderGANInterpol' --data 'C10' --noise 'Gaussian' --reals_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/CIFAR10/15052022_Base_non_para_cifar10_dcgan_WGAN_R1_01/FID_reals_train' --fakes_dir '/mnt/sdb1/siddarth/NeurIPS22_SpiderGAN_Logs/CIFAR10/15052022_Base_gaussian_cifar10_dcgan_WGAN_R1_01/FID_interpol_model_metrics' --order 1 --GPU '3' --SID_flag 0 --FID_flag 1 --KID_flag 1 --metric_mode 'clean'


###------ ###------ ###------ ###------ ###------ 

# python ./sid.py --model 'SpiderPGGAN' --data 'C10' --noise 'C10_Wts' --reals_dir '/raid/home/Siddarth/TFCodes/AlienCodes/ops/SpiderPGGAN/datasets/C10_RefImages/DID_tar_metrics' --fakes_dir '/raid/home/Siddarth/TFCodes/AlienCodes/ops/SpiderPGGAN/results/077-pgan-cifar10-NoiseTINWeightsTIN-preset-v2-8gpus-fp32/FinalImages' --order -1 --GPU '2' --FID_flag 1 --KID_flag 1 --SID_flag 0 --metric_mode 'legacy_tensorflow'


# python ./sid.py --model 'SpiderPGGAN' --data 'MetFaces' --noise 'C10' --reals_dir '/raid/home/Siddarth/TFCodes/AlienCodes/data/MetFaces/images' --fakes_dir '/raid/home/Siddarth/TFCodes/AlienCodes/ops/SpiderPGGAN/results/075-pgan-MetFaces-NoiseCIFAR10-preset-v2-8gpus-fp32/FinalImages' --order -1 --GPU '2' --FID_flag 1 --KID_flag 1 --SID_flag 0 --metric_mode 'legacy_tensorflow'


# python ./sid.py --model 'SpiderPGGAN' --data 'UkiyoE' --noise 'C10' --reals_dir '/raid/home/Siddarth/TFCodes/AlienCodes/data/UkiyoE/ukiyoe-1024' --fakes_dir '/raid/home/Siddarth/TFCodes/AlienCodes/ops/SpiderPGGAN/results/076-pgan-ukiyoe-NoiseCIFAR10-preset-v2-8gpus-fp32/FinalImages' --order -1 --GPU '2' --FID_flag 1 --KID_flag 1 --SID_flag 0 --metric_mode 'legacy_tensorflow'
### ------


# python ./sid.py --model 'SpiderGAN' --data 'MNIST' --noise 'MNIST_Four' --reals_dir '/home/siddarth/TF_Codes/AlienCodes/logs/Log_Folder_25052022/25052022_SpiderGAN_zeros_fmnist_mnist_deepconv_WGAN_R1_01/FID_reals_metrics' --fakes_dir '/home/siddarth/TF_Codes/AlienCodes/logs/Log_Folder_25052022/25052022_SpiderGAN_zeros_fmnist_mnist_deepconv_WGAN_R1_02/FID_fakes_metrics' --order 1 --GPU '0' --FID_flag 0 --KID_flag 0 --metric_mode 'clean'

# python ./sid.py --model 'SpiderGAN' --data 'MNIST' --noise 'MNIST_Round' --reals_dir '/home/siddarth/TF_Codes/AlienCodes/logs/Log_Folder_25052022/25052022_SpiderGAN_zeros_fmnist_mnist_deepconv_WGAN_R1_01/FID_reals_metrics' --fakes_dir '/home/siddarth/TF_Codes/AlienCodes/logs/Log_Folder_25052022/25052022_SpiderGAN_zeros_fmnist_mnist_deepconv_WGAN_R1_03/FID_fakes_metrics' --order 1 --GPU '1' --FID_flag 0 --KID_flag 0 --metric_mode 'clean'

# python ./sid.py --model 'SpiderGAN' --data 'MNIST' --noise 'MNIST_Full' --reals_dir '/home/siddarth/TF_Codes/AlienCodes/logs/Log_Folder_25052022/25052022_SpiderGAN_zeros_fmnist_mnist_deepconv_WGAN_R1_01/FID_reals_metrics' --fakes_dir '/home/siddarth/TF_Codes/AlienCodes/logs/Log_Folder_25052022/25052022_SpiderGAN_zeros_fmnist_mnist_deepconv_WGAN_R1_01/FID_fakes_metrics' --order 1 --GPU '1' --FID_flag 0 --KID_flag 0 --metric_mode 'clean'


# python ./sid.py --model 'SpiderGAN' --data 'MNIST_Full' --noise 'MNIST_Round' --reals_dir '/home/siddarth/TF_Codes/AlienCodes/logs/Log_Folder_25052022/25052022_SpiderGAN_zeros_fmnist_mnist_deepconv_WGAN_R1_01/FID_fakes_metrics' --fakes_dir '/home/siddarth/TF_Codes/AlienCodes/logs/Log_Folder_25052022/25052022_SpiderGAN_zeros_fmnist_mnist_deepconv_WGAN_R1_03/FID_fakes_metrics' --order 1 --GPU '1' --FID_flag 0 --KID_flag 0 --metric_mode 'clean'

# python ./sid.py --model 'SpiderGAN' --data 'MNIST_Full' --noise 'MNIST_Four' --reals_dir '/home/siddarth/TF_Codes/AlienCodes/logs/Log_Folder_25052022/25052022_SpiderGAN_zeros_fmnist_mnist_deepconv_WGAN_R1_01/FID_fakes_metrics' --fakes_dir '/home/siddarth/TF_Codes/AlienCodes/logs/Log_Folder_25052022/25052022_SpiderGAN_zeros_fmnist_mnist_deepconv_WGAN_R1_02/FID_fakes_metrics' --order 1 --GPU '1' --FID_flag 0 --KID_flag 0 --metric_mode 'clean'

# python ./sid.py --model 'SpiderGAN' --data 'MNIST_Round' --noise 'MNIST_Four' --reals_dir '/home/siddarth/TF_Codes/AlienCodes/logs/Log_Folder_25052022/25052022_SpiderGAN_zeros_fmnist_mnist_deepconv_WGAN_R1_03/FID_fakes_metrics' --fakes_dir '/home/siddarth/TF_Codes/AlienCodes/logs/Log_Folder_25052022/25052022_SpiderGAN_zeros_fmnist_mnist_deepconv_WGAN_R1_02/FID_fakes_metrics' --order 1 --GPU '1' --FID_flag 0 --KID_flag 0 --metric_mode 'clean'

###------ NeurIPS 2022 - SpiderGAN and SID ------###


# python ./sid.py --model 'StyleGAN3' --data 'AFHQ' --noise 'Gaussian' --reals_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --fakes_dir '/home/siddarth/TF_Codes/GitClones/stylegan3/StyleGAN2_AFHQ_512/' --order -1

# python ./sid.py --model 'StyleGAN2' --data 'AFHQ' --noise 'Gaussian' --reals_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --fakes_dir '/home/siddarth/TF_Codes/GitClones/stylegan3/StyleGAN3_AFHQ_512/' --order -1

# python ./sid.py --model 'none' --data 'AFHQ' --noise 'AFHQ' --reals_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --fakes_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --order -1

# ###------ m = 1

# python ./sid.py --model 'StyleGAN3' --data 'AFHQ' --noise 'Gaussian' --reals_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --fakes_dir '/home/siddarth/TF_Codes/GitClones/stylegan3/StyleGAN2_AFHQ_512/' --order 1 --GPU '0' --SID_flag 1 --FID_flag 1 --KID_flag 1 --metric_mode 'clean'

# python ./sid.py --model 'StyleGAN2' --data 'AFHQ' --noise 'Gaussian' --reals_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --fakes_dir '/home/siddarth/TF_Codes/GitClones/stylegan3/StyleGAN2_AFHQ_512/' --order 1 --GPU '0' --SID_flag 1 --FID_flag 1 --KID_flag 1 --metric_mode 'clean'

# python ./sid.py --model 'DataSet' --data 'AFHQ' --noise 'AFHQ' --reals_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --fakes_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --order 1 --GPU '0' --SID_flag 1 --FID_flag 1 --KID_flag 1 --metric_mode 'clean'

# ###------ m = -1

# python ./sid.py --model 'StyleGAN3' --data 'AFHQ' --noise 'Gaussian' --reals_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --fakes_dir '/home/siddarth/TF_Codes/GitClones/stylegan3/StyleGAN2_AFHQ_512/' --order -1 --GPU '0' --SID_flag 1 --FID_flag 0 --KID_flag 0 --metric_mode 'clean'

# python ./sid.py --model 'StyleGAN2' --data 'AFHQ' --noise 'Gaussian' --reals_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --fakes_dir '/home/siddarth/TF_Codes/GitClones/stylegan3/StyleGAN2_AFHQ_512/' --order -1 --GPU '0' --SID_flag 1 --FID_flag 0 --KID_flag 0 --metric_mode 'clean'

# python ./sid.py --model 'DataSet' --data 'AFHQ' --noise 'AFHQ' --reals_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --fakes_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --order -1 --GPU '0' --SID_flag 1 --FID_flag 0 --KID_flag 0 --metric_mode 'clean'

# ###------ m = -3

# python ./sid.py --model 'StyleGAN3' --data 'AFHQ' --noise 'Gaussian' --reals_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --fakes_dir '/home/siddarth/TF_Codes/GitClones/stylegan3/StyleGAN2_AFHQ_512/' --order -3 --GPU '0' --SID_flag 1 --FID_flag 0 --KID_flag 0 --metric_mode 'clean'

# python ./sid.py --model 'StyleGAN2' --data 'AFHQ' --noise 'Gaussian' --reals_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --fakes_dir '/home/siddarth/TF_Codes/GitClones/stylegan3/StyleGAN2_AFHQ_512/' --order -3 --GPU '0' --SID_flag 1 --FID_flag 0 --KID_flag 0 --metric_mode 'clean'

# python ./sid.py --model 'DataSet' --data 'AFHQ' --noise 'AFHQ' --reals_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --fakes_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --order -3 --GPU '0' --SID_flag 1 --FID_flag 0 --KID_flag 0 --metric_mode 'clean'

# ###------ m = -5

# python ./sid.py --model 'StyleGAN3' --data 'AFHQ' --noise 'Gaussian' --reals_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --fakes_dir '/home/siddarth/TF_Codes/GitClones/stylegan3/StyleGAN2_AFHQ_512/' --order -5 --GPU '0' --SID_flag 1 --FID_flag 0 --KID_flag 0 --metric_mode 'clean'

# python ./sid.py --model 'StyleGAN2' --data 'AFHQ' --noise 'Gaussian' --reals_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --fakes_dir '/home/siddarth/TF_Codes/GitClones/stylegan3/StyleGAN2_AFHQ_512/' --order -5 --GPU '0' --SID_flag 1 --FID_flag 0 --KID_flag 0 --metric_mode 'clean'

# python ./sid.py --model 'DataSet' --data 'AFHQ' --noise 'AFHQ' --reals_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --fakes_dir '/home/siddarth/TF_Codes/GitClones/stargan-v2/data/afhq/train/all' --order -5 --GPU '0' --SID_flag 1 --FID_flag 0 --KID_flag 0 --metric_mode 'clean'

