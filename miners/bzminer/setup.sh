# miners/bzminer/setup.sh
#!/bin/bash
cd "$(dirname "$0")"
wget https://github.com/bzminer/bzminer/releases/download/v24.0.2/bzminer_v24.0.2_linux.tar.gz
tar -xvf bzminer_v24.0.2_linux.tar.gz
mv bzminer_v24.0.2_linux/* .
chmod +x bzminer
rm -rf bzminer_v24.0.2_linux bzminer_v24.0.2_linux.tar.gz
