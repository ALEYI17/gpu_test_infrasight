# miners/nbminer/setup.sh
#!/bin/bash
cd "$(dirname "$0")"
wget https://github.com/NebuTech/NBMiner/releases/download/v42.3/NBMiner_42.3_Linux.tgz
tar -xvf NBMiner_42.3_Linux.tgz
mv NBMiner_Linux/* .
chmod +x nbminer
rm -rf NBMiner_Linux NBMiner_42.3_Linux.tgz
