import matplotlib.pyplot as plt
import math
import numpy as np

# --- SENİN KODUNDAKİ AYARLAR ---
EPS_START = 1.0     # Başlangıçta %100 Rastgele
EPS_END = 0.01      # En sonunda %1 Rastgele
EPS_DECAY = 2000    # Senin belirlediğin azalma hızı

# Simülasyon: 0'dan 15.000 adıma kadar (Çünkü ajan geliştikçe oyun uzayacak)
# Senin kodunda 'steps_done' her karede (frame) artıyor.
total_steps = 15000 
steps = range(total_steps)
epsilon_values = []

# Senin kodundaki formülün aynısı:
for step in steps:
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * step / EPS_DECAY)
    epsilon_values.append(eps_threshold)

# --- GRAFİK ÇİZİMİ ---
plt.figure(figsize=(10, 6))

# Eğriyi çiz
plt.plot(steps, epsilon_values, color='#2ecc71', linewidth=2.5, label='Epsilon (Rastgelelik)')

# Kritik Nokta: 2000. Adım (Senin Decay ayarın)
decay_val = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * EPS_DECAY / EPS_DECAY)
plt.plot(EPS_DECAY, decay_val, 'ro', markersize=8) # Kırmızı nokta
plt.text(EPS_DECAY + 500, decay_val, f'Decay Noktası\n(Adım: {EPS_DECAY}, Eps: {decay_val:.2f})', color='red')

# Kritik Nokta: Eğitim Sonu (Tahmini)
# Ajan iyi oynarsa 600 bölüm ortalama 100-200 adım sürse toplam ~60.000 adım eder.
# Biz grafikte ilk 15.000 adımı gösteriyoruz ki değişim net görülsün.

plt.title(f'Senin Ajanının Öğrenme Stratejisi\n(EPS_DECAY = {EPS_DECAY})', fontsize=14, fontweight='bold')
plt.xlabel('Toplam Atılan Adım Sayısı (steps_done)', fontsize=12)
plt.ylabel('Rastgele Hareket Etme İhtimali', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend()

# Alanı boya (Görsellik)
plt.fill_between(steps, epsilon_values, EPS_END, alpha=0.1, color='#2ecc71')

plt.tight_layout()
plt.show()