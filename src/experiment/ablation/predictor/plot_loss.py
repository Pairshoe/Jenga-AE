import json
import matplotlib.pyplot as plt
  
loss_llama2_la = []
loss_llama2_rd = []
loss_opt_la = []
loss_opt_rd = []

with open('logs/ablations/predictor/llama2_16384_predictor_red/trainer_state.json', 'r') as f:
    loss_llama2_la = [entry['loss'] for entry in json.load(f)['log_history'] if 'loss' in entry]
with open('logs/ablations/predictor/llama2_16384_predictor_la/trainer_state.json', 'r') as f:
    loss_llama2_rd = [entry['loss'] for entry in json.load(f)['log_history'] if 'loss' in entry]
with open('logs/ablations/predictor/opt-6.7b_16384_predictor_red/trainer_state.json', 'r') as f:
    loss_opt_la =[entry['loss'] for entry in json.load(f)['log_history'] if 'loss' in entry]
with open('logs/ablations/predictor/opt-6.7b_16384_predictor_la/trainer_state.json', 'r') as f:
    loss_opt_rd = [entry['loss'] for entry in json.load(f)['log_history'] if 'loss' in entry]
  
print('len of loss_llama2_la:', len(loss_llama2_la))
print('len of loss_llama2_rd:', len(loss_llama2_rd))
print('len of loss_opt_la:', len(loss_opt_la))
print('len of loss_opt_rd:', len(loss_opt_rd))

len_limit = 400
loss_llama2_la = loss_llama2_la[:len_limit]
loss_llama2_rd = loss_llama2_rd[:len_limit]
loss_opt_la = loss_opt_la[:len_limit]
loss_opt_rd = loss_opt_rd[:len_limit]

plt.figure(figsize=(4, 3))

colors = ['#255475', '#5D7F84', '#DCBCAC', '#D6838D', '#F3AE75', '#F8F1E4']

plt.plot(loss_llama2_la, color=colors[0], linewidth=2)
plt.plot(loss_llama2_rd, color=colors[1], linewidth=2)
plt.plot(loss_opt_la, color=colors[2], linewidth=2)
plt.plot(loss_opt_rd, color=colors[3], linewidth=2)

plt.grid('y', linestyle='--', alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
 
plt.tight_layout()
plt.savefig('output_figures/ablations/predictor/exp-ablation-predictor-loss.pdf', bbox_inches='tight')
