# miners/setup.sh
#!/bin/bash
echo "Setting up all miners..."
for dir in nbminer gminer bzminer srbminer t-rex xmrig lolminer; do
  if [ -f "miners/$dir/setup.sh" ]; then
    echo "Setting up $dir..."
    bash $dir/setup.sh
  fi
done
echo "All miners ready!"
