git config --global user.email "gertelrina@yandex.ru"
git config --global user.name "agertel"

# wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run

# apt-get --purge remove cuda nvidia* libnvidia-*
# dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 dpkg --purge
# apt-get remove cuda-*
# apt autoremove
# apt-get update

# sudo sh cuda_11.7.0_515.43.04_linux.run  # wo driver

# sudo apt install ubuntu-drivers-common
# sudo ubuntu-drivers autoinstall
# sudo apt install nvidia-driver-470
#  # 79 # 

#  nvcc --version