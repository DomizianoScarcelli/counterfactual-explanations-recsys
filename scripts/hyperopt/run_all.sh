#!/bin/bash

SESSION_NAME="full_eval"

# Function to check if a Bash script is already running inside tmux
is_running_in_tmux() {
    local script_name=$1
    tmux list-panes -a -F "#{pane_pid}" | while read -r pane_pid; do
        if pgrep -P "$pane_pid" -f "bash $script_name" > /dev/null; then
            return 0  # Bash script is running
        fi
        if pgrep -P "$pane_pid" -f "python" > /dev/null; then
            return 0  # Python script inside Bash is running
        fi
    done
    return 1  # Script is NOT running
}

# Start a new tmux session if it doesn't exist
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_NAME"
fi

# Function to run a Bash script in a new tmux tab if it's not already running
run_script() {
    local script_name=$1
    local window_name=$2

    if is_running_in_tmux "$script_name"; then
        echo "Skipping $script_name (already running)"
    else
        tmux new-window -t "$SESSION_NAME" -n "$window_name"
        tmux send-keys -t "$SESSION_NAME":"$window_name" "bash $script_name" C-m
    fi
}

# Run multiple Bash scripts (each containing a Python script inside)
run_script "scripts/hyperopt.sh" "Script1"

# Attach to the tmux session
tmux attach-session -t "$SESSION_NAME"
