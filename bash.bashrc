# If not running interactively, don't do anything
[ -z "$PS1" ] && return

export TERM=xterm-256color

# Useful aliases
alias ls="ls --color=auto"
alias ll="ls -l"
alias aws='aws --endpoint-url http://tobiasnorlund.mine.nu:9000'