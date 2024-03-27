#!/bin/bash

echo "[1/4] Running Q-learning..."
#echo "Env: FrozenLake (episodic)" && python3 run_experiment.py --agentfile qlearner.py
#echo "Env: FrozenLake (episodic)" && python3 run_experiment.py --agentfile qlearner.py --init low
#echo "Env: FrozenLake (episodic)" && python3 run_experiment.py --agentfile qlearner.py --init high

#echo "Env: RiverSwim (continuous)" && python3 run_experiment.py --agentfile qlearner.py --env riverswim:RiverSwim
echo "Env: RiverSwim (continuous)" && python3 run_experiment.py --agentfile qlearner.py --env riverswim:RiverSwim --init low
echo "Env: RiverSwim (continuous)" && python3 run_experiment.py --agentfile qlearner.py --env riverswim:RiverSwim --init high

echo -e "\n[2/4] Running double Q-learning..."
#echo "Env: FrozenLake (episodic)" && python3 run_experiment.py --agentfile dqlearner.py
#echo "Env: FrozenLake (episodic)" && python3 run_experiment.py --agentfile dqlearner.py --init low
#echo "Env: FrozenLake (episodic)" && python3 run_experiment.py --agentfile dqlearner.py --init high

#echo "Env: RiverSwim (continuous)" && python3 run_experiment.py --agentfile dqlearner.py --env riverswim:RiverSwim
echo "Env: RiverSwim (continuous)" && python3 run_experiment.py --agentfile dqlearner.py --env riverswim:RiverSwim --init low
echo "Env: RiverSwim (continuous)" && python3 run_experiment.py --agentfile dqlearner.py --env riverswim:RiverSwim --init high

echo -e "\n[3/4] Running SARSA..."
#echo "Env: FrozenLake (episodic)" && python3 run_experiment.py --agentfile sarsa.py
#echo "Env: FrozenLake (episodic)" && python3 run_experiment.py --agentfile sarsa.py --init low
#echo "Env: FrozenLake (episodic)" && python3 run_experiment.py --agentfile sarsa.py --init high

#echo "Env: RiverSwim (continuous)" && python3 run_experiment.py --agentfile sarsa.py --env riverswim:RiverSwim
echo "Env: RiverSwim (continuous)" && python3 run_experiment.py --agentfile sarsa.py --env riverswim:RiverSwim --init low
echo "Env: RiverSwim (continuous)" && python3 run_experiment.py --agentfile sarsa.py --env riverswim:RiverSwim --init high

echo -e "\n[4/4] Running expected SARSA..."
#echo "Env: FrozenLake (episodic)" && python3 run_experiment.py --agentfile esarsa.py
#echo "Env: FrozenLake (episodic)" && python3 run_experiment.py --agentfile esarsa.py --init low
#echo "Env: FrozenLake (episodic)" && python3 run_experiment.py --agentfile esarsa.py --init high

#echo "Env: RiverSwim (continuous)"  && python3 run_experiment.py --agentfile esarsa.py --env riverswim:RiverSwim
echo "Env: RiverSwim (continuous)"  && python3 run_experiment.py --agentfile esarsa.py --env riverswim:RiverSwim --init low
echo "Env: RiverSwim (continuous)"  && python3 run_experiment.py --agentfile esarsa.py --env riverswim:RiverSwim --init high
