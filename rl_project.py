import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 1. ORTAM AYARLARI
env = gym.make("CartPole-v1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. YAPAY SİNİR AĞI (BEYİN)
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# 3. HİPERPARAMETRELER - DÜZELTİLMİŞ
BATCH_SIZE = 64        # Daha küçük batch (daha sık güncelleme)
GAMMA = 0.99           
EPS_START = 1.0        # %100 exploration'dan başla
EPS_END = 0.01         # Daha düşük minimum
EPS_DECAY = 2000       # Daha yavaş azalt (daha uzun exploration)
TAU = 0.005            
LR = 1e-3              # Daha yüksek learning rate

# 4. AKSİYON SEÇİMİ VE HAFIZA
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = deque(maxlen=10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = random.sample(memory, BATCH_SIZE)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
    
    batch_state = torch.cat(batch_state)
    batch_action = torch.cat(batch_action)
    batch_reward = torch.cat(batch_reward)
    
    # ✅ DÜZELTİLDİ: None değerleri sıfır tensor'a çevir
    batch_next_state = torch.cat([s if s is not None else torch.zeros(1, n_observations, device=device) 
                                  for s in batch_next_state])
    batch_done = torch.tensor(batch_done, dtype=torch.float32, device=device)

    current_q_values = policy_net(batch_state).gather(1, batch_action)
    next_q_values = target_net(batch_next_state).max(1)[0].detach()
    
    # ✅ DÜZELTİLDİ: Oyun bitince gelecek ödül 0 olacak (1 - batch_done ile)
    expected_q_values = batch_reward + (GAMMA * next_q_values * (1 - batch_done))

    criterion = nn.SmoothL1Loss()
    loss = criterion(current_q_values, expected_q_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# 5. EĞİTİM DÖNGÜSÜ
num_episodes = 600
print("Eğitim Başlıyor...")
episode_durations = []

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    episode_reward = 0
    for t in range(500):
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        
        # ✅ DÜZELTİLDİ: Oyun erken biterse ceza ver!
        if terminated and t < 499:
            reward = -10  # Çubuk düşerse CEZA
        
        reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
        done = terminated or truncated
        episode_reward += reward

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # ✅ DÜZELTİLDİ: Her durumda hafızaya kaydet (oyun bitince de!)
        memory.append((state, action, reward_tensor, next_state, done))
        
        state = next_state
        optimize_model()

        # Hedef ağı güncelle
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            if (i_episode + 1) % 50 == 0:
                avg_score = sum(episode_durations[-50:]) / min(50, len(episode_durations))
                print(f"Bölüm {i_episode+1} | Skor: {t+1} | Ortalama (son 50): {avg_score:.1f}")
            break

print("Eğitim Tamamlandı!")

# Son 100 bölümün ortalamasını göster
if len(episode_durations) >= 100:
    final_avg = sum(episode_durations[-100:]) / 100
    print(f"\n🎯 Son 100 bölüm ortalaması: {final_avg:.1f}")

# Grafik çiz
plt.figure(figsize=(10, 5))
plt.plot(episode_durations, alpha=0.6, linewidth=0.5)
# Hareketli ortalama
window = 50
if len(episode_durations) >= window:
    moving_avg = [sum(episode_durations[i:i+window])/window 
                  for i in range(len(episode_durations)-window+1)]
    plt.plot(range(window-1, len(episode_durations)), moving_avg, 
             'r-', linewidth=2, label=f'{window}-Bölüm Ortalaması')
plt.xlabel('Bölüm')
plt.ylabel('Skor (Süre)')
plt.title('DQN Öğrenme Eğrisi - CartPole')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cartpole_learning_curve.png', dpi=150)
print("\n📊 Grafik 'cartpole_learning_curve.png' olarak kaydedildi!")

# 6. EĞİTİLMİŞ MODELİ İZLEME
env.close()
env = gym.make("CartPole-v1", render_mode="human")

print("\n🎮 Eğitilmiş model çalışıyor... (Kapatmak için Ctrl+C)")
state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
test_scores = []
current_score = 0

try:
    while True:
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)
        
        observation, reward, terminated, truncated, _ = env.step(action.item())
        current_score += 1
        
        if terminated or truncated:
            test_scores.append(current_score)
            print(f"Test Skoru: {current_score} | Ortalama: {sum(test_scores)/len(test_scores):.1f}")
            current_score = 0
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
except KeyboardInterrupt:
    print("\n\n✅ Test sonlandırıldı.")
    env.close()