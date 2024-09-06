pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

sudo apt install pkg-config
sudo apt install libssl-dev
nvm install 20
nvm use 20

git config push.autoSetupRemote true
git config --global --add safe.directory /workspaces/cloud-infrastructure
git clone https://github.com/RuiFilipeCampos/nvim.git "${XDG_CONFIG_HOME:-$HOME/.config}"/nvim
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
$HOME/.fzf/install --key-bindings --completion --update-rc
echo "alias nf='nvim \$(fzf)'" >> $HOME/.bashrc
nvim --headless "+Lazy! sync" +qa 


echo "eval \$(op signin )" >> $HOME/.bashrc
echo "op run -- gh auth setup-git" >> $HOME/.bashrc
echo "echo 'Running git setup'" >> $HOME/.bashrc

echo "git config --global user.email \$(op read op://digital-defiance-personal/github/GH_EMAIL)" >> $HOME/.bashrc
echo "git config --global user.name \$(op read op://digital-defiance-personal/github/GH_NAME)" >> $HOME/.bashrc
