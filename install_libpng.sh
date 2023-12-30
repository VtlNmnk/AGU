sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install --force-yes -y libsdl2-image-2.0-0 libsdl2-image-dev yasm
sudo apt-get remove libnuma-dev
sudo apt-get install -y libsdl-image1.2-dev libsdl1.2-dev libjpeg8-dev emscripten
cd ..
wget https://sourceforge.net/projects/libpng/files/libpng16/1.6.37/libpng-1.6.37.tar.gz
tar xzf libpng-1.6.37.tar.gz
cd libpng-1.6.37/
./configure --prefix=/usr --disable-static && make
make check
sudo make install
dpkg -l | grep libpng

