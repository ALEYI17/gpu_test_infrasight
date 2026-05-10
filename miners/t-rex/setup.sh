# miners/t-rex/setup.sh
#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Downloading T-Rex..."
wget https://github.com/trexminer/T-Rex/releases/download/0.26.8/t-rex-0.26.8-linux.tar.gz
tar -xvf t-rex-0.26.8-linux.tar.gz
chmod +x t-rex
rm t-rex-0.26.8-linux.tar.gz

echo "T-Rex ready!"
ls -la t-rex
