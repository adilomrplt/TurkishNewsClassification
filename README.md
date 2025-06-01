# Türkçe Haber Sınıflandırma - Makine Öğrenmesi Projesi

Bu proje, Türkçe haber metinlerini otomatik olarak sınıflandırmak için geliştirilmiş bir makine öğrenmesi uygulamasıdır. Farklı sınıflandırma algoritmalarını kullanarak haber metinlerinin kategorilerini tahmin eder.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/adilomrplt/TurkishNewsClassification)

## Proje Hakkında

### Özellikler
- Türkçe haber metinlerini otomatik sınıflandırma
- Çoklu sınıflandırma algoritması desteği
- TF-IDF vektörizasyonu ile metin özellik çıkarımı
- Grid Search ile hiperparametre optimizasyonu
- Model performans karşılaştırması
- Eğitilmiş modellerin kaydedilmesi ve yüklenmesi
- Türkçe stop words desteği
- N-gram özellik çıkarımı (1-gram ve 2-gram)

### Kullanılan Teknolojiler
- Python 3.x
- scikit-learn (makine öğrenmesi kütüphanesi)
- NumPy (sayısal işlemler)
- JSON (veri depolama)

### Proje Yapısı
```
TurkishNewsClassification/
├── data/                     # Veri seti dizini
│   └── news.json            # Haber verileri
├── models/                   # Eğitilmiş modeller
├── run.py                   # Ana uygulama dosyası
├── train.py                 # Model eğitimi
├── classification.py        # Sınıflandırma algoritmaları
├── data.py                  # Veri yönetimi
├── nlp.py                   # Doğal dil işleme
├── vectorization.py         # Metin vektörizasyonu
├── requirements.txt         # Proje bağımlılıkları
├── .gitignore              # Git tarafından yok sayılacak dosyalar
└── README.md               # Proje dokümantasyonu
```

### Desteklenen Sınıflandırma Algoritmaları
- Decision Tree (Karar Ağacı)
- K-Nearest Neighbors (KNN)
- Stochastic Gradient Descent (SGD)
- Naive Bayes
- Support Vector Machines (SVM)
- Random Forest
- Logistic Regression

### Veri Yapısı
Her haber kaydı aşağıdaki bilgileri içerir:
- Metin: Haber içeriği
- Skor: Haberin 0-10 arası puanı (veya duygu etiketi)
- Vektör: TF-IDF ile oluşturulmuş özellik vektörü


### Kurulum ve Çalıştırma
1. Python 3.x sürümünün yüklü olduğundan emin olun.
2. Projeyi klonlayın:
   ```bash
   git clone https://github.com/adilomrplt/TurkishNewsClassification.git
   cd TurkishNewsClassification
   ```
3. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
4. Model eğitimi için:
   ```bash
   python run.py
   ```

### Model Eğitimi ve Değerlendirme
- Her algoritma için ayrı grid search parametreleri
- Çapraz doğrulama ile model değerlendirmesi
- En iyi modelin otomatik seçimi ve kaydedilmesi
- Performans metriklerinin hesaplanması

### Hiperparametre Optimizasyonu
Her algoritma için optimize edilen parametreler:
- TF-IDF parametreleri (max_features, ngram_range)
- Algoritma-spesifik parametreler (örn. KNN için n_neighbors, SVM için C ve kernel)
- Düzenleme parametreleri (penalty, alpha)

### Katkıda Bulunma
1. Bu repository'yi fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik: Açıklama'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Bir Pull Request oluşturun


### İletişim
- GitHub: [@adilomrplt](https://github.com/adilomrplt)
- Repository: [TurkishNewsClassification](https://github.com/adilomrplt/TurkishNewsClassification) 