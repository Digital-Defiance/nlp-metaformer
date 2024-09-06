nvm install 20
nvm use 20

git config push.autoSetupRemote true
git config --global --add safe.directory /workspaces/cloud-infrastructure
git clone https://github.com/RuiFilipeCampos/nvim.git "${XDG_CONFIG_HOME:-$HOME/.config}"/nvim
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
$HOME/.fzf/install --key-bindings --completion --update-rc
echo "alias nf='nvim \$(fzf)'" >> $HOME/.bashrc
nvim --headless "+Lazy! sync" +qa 
