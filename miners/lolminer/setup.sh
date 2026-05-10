# miners/lolminer/setup.sh
#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Downloading lolMiner..."
wget https://github.com/Lolliedieb/lolMiner-releases/releases/download/1.98a/lolMiner_v1.98a_Lin64.tar.gz
tar -xvf lolMiner_v1.98a_Lin64.tar.gz
mv 1.98a/* .
chmod +x lolMiner
rm -rf 1.98a lolMiner_v1.98a_Lin64.tar.gz

echo "lolMiner ready!"
ls -la lolMiner
