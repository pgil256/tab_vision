# TabVision Agent Farm Setup Guide

## Prerequisites

```bash
# Clone the Claude Code Agent Farm (if not already done)
git clone https://github.com/Dicklesworthstone/claude_code_agent_farm.git
cd claude_code_agent_farm

# Install dependencies
pip install -r requirements.txt
```

## Setup TabVision for Agent Farm

```bash
# Navigate to TabVision project
cd /home/gilhooleyp/projects/tab_vision

# Ensure coordination directory exists
mkdir -p coordination/agent_locks

# Create a new branch for agent work
git checkout -b agent-farm-improvements

# Install project dependencies (both frontend and backend)
cd tabvision-client && npm install && cd ..
cd tabvision-server && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && cd ..
```

## Run the Agent Farm

### Option 1: From Agent Farm Directory

```bash
# Navigate to agent farm
cd /path/to/claude_code_agent_farm

# Run with TabVision config
python run_agent_farm.py \
  --config /home/gilhooleyp/projects/tab_vision/tabvision_agent_farm_config.json \
  --prompt /home/gilhooleyp/projects/tab_vision/tabvision_agent_farm_prompt.txt \
  --workdir /home/gilhooleyp/projects/tab_vision
```

### Option 2: Copy Config to Agent Farm

```bash
# Copy config files to agent farm configs directory
cp /home/gilhooleyp/projects/tab_vision/tabvision_agent_farm_config.json /path/to/claude_code_agent_farm/configs/
cp /home/gilhooleyp/projects/tab_vision/tabvision_agent_farm_prompt.txt /path/to/claude_code_agent_farm/prompts/

# Run from agent farm directory
cd /path/to/claude_code_agent_farm
python run_agent_farm.py --config configs/tabvision_agent_farm_config.json
```

### Option 3: Using tmux Directly

```bash
# Start a new tmux session
tmux new-session -s tabvision_agents

# Split into 9 panes (for 9 agents)
# Horizontal splits
tmux split-window -h
tmux split-window -h
tmux select-layout even-horizontal

# Vertical splits in each
tmux select-pane -t 0 && tmux split-window -v && tmux split-window -v
tmux select-pane -t 3 && tmux split-window -v && tmux split-window -v
tmux select-pane -t 6 && tmux split-window -v && tmux split-window -v

# In each pane, run claude with the agent prompt
# Pane 0 (Agent 1 - Audio Pipeline):
claude --prompt "You are Agent 1: Audio Pipeline Specialist. $(cat tabvision_agent_farm_prompt.txt)"

# Repeat for other agents with their specific role...
```

## Monitor Agent Progress

```bash
# Watch the coordination files
watch -n 5 cat coordination/active_work_registry.json

# View completed work
cat coordination/completed_work_log.json | jq .

# Check for lock files (active agents)
ls -la coordination/agent_locks/

# View tmux session
tmux attach -t tabvision_agents
```

## Useful tmux Commands

```bash
# List sessions
tmux ls

# Attach to session
tmux attach -t tabvision_agents

# Kill session when done
tmux kill-session -t tabvision_agents

# Switch between panes
Ctrl+b, arrow keys

# Scroll in a pane
Ctrl+b, [    # Enter scroll mode
q            # Exit scroll mode
```

## Verify Agent Work

```bash
# Check git status for changes
git status

# View recent commits
git log --oneline -20

# Run tests to verify nothing is broken
cd tabvision-server && pytest tests/ -v
cd tabvision-client && npm run typecheck
```

## Cleanup After Agent Run

```bash
# Clear coordination files
echo '{"registry_version":"1.0","last_updated":null,"claimed_work":{},"agents_active":[]}' > coordination/active_work_registry.json
rm -f coordination/agent_locks/*.lock

# Review and commit agent work
git add -A
git status
git commit -m "Agent farm improvements: [describe changes]"
```

## Troubleshooting

```bash
# If agents conflict on same files, check locks
cat coordination/agent_locks/*.lock

# Reset coordination state
rm -rf coordination/agent_locks/*
echo '{}' > coordination/active_work_registry.json

# Check for merge conflicts
git diff --check

# If backend tests fail
cd tabvision-server
source venv/bin/activate
pytest tests/ -v --tb=long

# If frontend build fails
cd tabvision-client
npm run typecheck
npm run build
```
