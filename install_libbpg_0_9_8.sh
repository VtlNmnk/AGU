sudo apt-get install -y yasm
sudo apt-get install -y cmake
sudo apt install -y git
cd ..
rm -rf libbpg
git clone https://github.com/mirrorer/libbpg
cd libbpg/
make install
sudo checkinstall
sudo ldconfig /usr/local/lib
bpgenc
cd ..

