# RL CartPole-v1 DQN Solver 🕹️

Bu proje, OpenAI Gymnasium'un klasik **CartPole-v1** ortamındaki denge problemini çözmek için **Deep Q-Learning (DQN)** algoritmasını kullanır. Proje, PyTorch kütüphanesi üzerine inşa edilmiştir ve ajanın öğrenme sürecini görselleştirmek için grafikler oluşturur.

##  Proje Hakkında

Ajan (pole), bir araba üzerindeki çubuğu devirmemeye çalışır. Araba sağa veya sola hareket ettirilerek denge sağlanır.

* **Durum (State):** Arabanın konumu, hızı, çubuğun açısı ve açısal hızı.
* **Aksiyon (Action):** Sola it (0) veya Sağa it (1).
* **Ödül (Reward):** Çubuğun dik durduğu her an için +1 ödül verilir. Çubuk düştüğünde ise bu modelde ekstra **-10 ceza** uygulanmaktadır.

##  Kurulum

Öncelikle gerekli kütüphaneleri bilgisayarınıza yüklemeniz gerekir:

```bash
pip install gymnasium[classic_control] torch matplotlib numpy

```

## 🧠 Model Mimarisi

Kullanılan DQN yapısı şu şekildedir:

* **Giriş Katmanı:** 4 (Ortam gözlem sayısı)
* **Gizli Katmanlar:** 2 adet 128 birimlik tam bağlı (Linear) katman ve ReLU aktivasyon fonksiyonları.
* **Çıkış Katmanı:** 2 (Her bir aksiyon için Q-değeri)

## 📈 Hiperparametreler

Modelin başarısında kritik rol oynayan değerler:
| Parametre | Değer | Açıklama |
| :--- | :--- | :--- |
| `BATCH_SIZE` | 64 | Her eğitim adımında hafızadan çekilen örnek sayısı |
| `GAMMA` | 0.99 | Gelecekteki ödüllerin önem derecesi (Discount Factor) |
| `LR` | 0.001 | Öğrenme oranı (AdamW Optimizer) |
| `EPS_DECAY` | 2000 | Exploration (keşif) oranının azalma hızı |
| `TAU` | 0.005 | Target ağının yumuşak güncelleme (Soft Update) oranı |

## 📊 Eğitim Süreci

Eğitim sırasında ajan önce rastgele hareketler yaparak çevreyi keşfeder (Exploration). Zamanla `epsilon` değerinin düşmesiyle birlikte öğrendiği bilgileri kullanmaya başlar (Exploitation).

Eğitim sonunda `cartpole_learning_curve.png` adında bir grafik oluşturulur. Bu grafik bölümlere göre alınan skorları ve 50 bölümlük hareketli ortalamayı gösterir.

## 🎮 Kullanım

Eğitimi başlatmak ve ardından eğitilmiş modeli izlemek için:

```bash
python main.py

```

Eğitim tamamlandıktan sonra otomatik olarak `render_mode="human"` açılacak ve ajanın çubuğu nasıl dengelediğini izleyebileceksiniz.

---

### Projeyi Push Etmek İçin Hatırlatma

`.venv` klasörünü daha önce temizlediğimiz için artık güvenle şu komutları kullanabilirsin:

```bash
git add .
git commit -m "feat: add cartpole dqn solver and readme"
git push origin main

```

---

Bu README dosyasına eklememi istediğin özel bir başlık veya görsel var mı? Eğer istersen eğitimden sonra oluşan grafiği de bu dosyaya gömecek şekilde güncelleyebiliriz.
