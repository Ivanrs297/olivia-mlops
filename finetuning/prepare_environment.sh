# remove conda environment
sudo conda remove -y --name pytorch --all

# update list of packages
sudo apt update

# install python
sudo apt install python-is-python3 -y

# install pip
sudo apt install python3-pip -y

# install jupyterlab
pip install jupyterlab

# update PATH
# nano ~/.bashrc
# export PATH="$PATH:/home/ubuntu/.local/bin"
# source ~/.bashrc

# install gh client
type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y

# authenticate in github
# gh auth login
# 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

# clone repo
gh repo clone davidlainesv/olivia-finetuning

# start jupyterlab
jupyter lab --ip 0.0.0.0 --port 1234 --allow-root --no-browser
