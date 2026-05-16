mkdir -p /home/hukkyu/new_folder/subfolder
cd /root/.vscode-server/data/User/History/2106bdf4

for f in *.ipynb; do
  if jq . "$f" > /home/hukkyu/new_folder/subfolder/restored_"$f" 2>/dev/null; then
    echo "✔ Restored $f"
  else
    echo "✘ Failed to restore $f"
  fi
done

