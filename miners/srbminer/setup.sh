# miners/srbminer/setup.sh
#!/bin/bash
cd "$(dirname "$0")"
wget https://github.com/doktor83/SRBMiner-Multi/releases/download/3.2.8/SRBMiner-Multi-3-2-8-Linux.tar.gz
tar -xvf SRBMiner-Multi-3-2-8-Linux.tar.gz
mv SRBMiner-Multi-3-2-8/* .
chmod +x SRBMiner-MULTI
rm -rf SRBMiner-Multi-3-2-8 SRBMiner-Multi-3-2-8-Linux.tar.gz
