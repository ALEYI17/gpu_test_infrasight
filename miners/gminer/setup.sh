# miners/gminer/setup.sh
#!/bin/bash
cd "$(dirname "$0")"
wget https://github.com/develsoftware/GMinerRelease/releases/download/3.44/gminer_3_44_linux64.tar.xz
tar -xvJf gminer_3_44_linux64.tar.xz
chmod +x miner
rm gminer_3_44_linux64.tar.xz
