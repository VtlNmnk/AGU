cd ..
rm -rf libbpg
mkdir libbpg
cd libbpg/
wget http://bellard.org/bpg/libbpg-0.9.5.tar.gz
tar xzf libbpg-0.9.5.tar.gz
cd libbpg-0.9.5/
make -j 4
sudo make install
sudo checkinstall
sudo ldconfig /usr/local/lib
bpgenc

