#! /bin/bash
function init_git_hooks() {
  export http_proxy=http://sys-proxy-rd-relay.byted.org:3128
  export https_proxy=http://sys-proxy-rd-relay.byted.org:3128
  export ftp_proxy=http://sys-proxy-rd-relay.byted.org:3128
  export no_proxy=.byted.org
  export PATH=$PATH:$HOME/.local/bin
  python3 -m pip config set global.disable-pip-version-check true
  [[ $(python3 -m pip list | grep -i pre-commit) ]] || [[ $(python3 -m pip install pre-commit) ]]
  [[ $(pre-commit install -f -t pre-commit -c .pre-commit-config.yaml) ]] && echo "initialize git pre-commit hooks successfully!"
  [[ $(pre-commit install -f -t commit-msg -c .commit-msg-config.yaml) ]] && echo "initialize git commit-msg hooks successfully!"
}
# initialize git hooks for the repository
init_git_hooks
