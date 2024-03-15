cd /home
sudo apt update
sudo apt install python3-pip
python3 --version
sudo apt-get install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
# sudo reboot
nvidia-smi
sudo wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh   # Modify this link to the latest version
bash Anaconda3-2024.02-1-Linux-x86_64.sh   # Modify the file name to the latest version
source ~/.bashrc
echo 'export PATH="/home/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
# conda init