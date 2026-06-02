# RiskRider

RiskRider, motosiklet sürücülerinin güvenliğini analiz eden, görüntü, canlı yayın ve video üzerinden risk skoru üreten yapay zeka destekli bir web uygulamasıdır. Sistem YOLO tespitlerini kullanır ve Streamlit arayüzü ile hızlı, anlaşılır ve akıcı bir deneyim sunar.

## Özellikler

### Görüntü Analizi

Tek bir fotoğraf yükleyerek sürücünün güvenlik durumunu anlık olarak değerlendirebilirsiniz. Sistem, tespit ettiği risklere göre skoru günceller ve sonuçları görselleştirir.

![Görüntü Analizi](img/görüntüAnalizi.png)

![Görüntü Analizi 2](img/GörüntüAnalizi-2.png)

### Canlı Yayın (IP Webcam / RTSP)

Canlı akış üzerinden gerçek zamanlı tespit yapılır. Akış dalgalanmalarında kare atlama mantığı ile sistem stabil kalır ve arayüz donmadan çalışır.

![Canlı Yayın](img/CanlıYayın.png)

### Video Analizi

MP4 video dosyaları kare kare analiz edilir. Zaman ekseninde risk skorunun nasıl değiştiğini grafikte takip edebilirsiniz.

![Video Analizi](img/VideoAnalizi.png)

### Tespitli Görüntü ve Detaylı Liste

Sistemin tespit ettiği her sınıf görsel üzerinde işaretlenir. Ayrıca tüm tespitler güven skorlarıyla birlikte listelenir.

### Risk Skoru ve Seviye Etiketleri

Skor 0-100 aralığında hesaplanır ve sonuç düşük, orta, yüksek gibi seviyelerle etiketlenir. Böylece çıktı tek bakışta anlaşılır.

## Kurulum

Sistem genelinde pip kurulumları PEP 668 sebebiyle engellenebileceği için sanal ortam (venv) önerilir.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Çalıştırma

```bash
python -m streamlit run app.py
```

Ardından tarayıcıda şu adresi açın:

```
http://localhost:8501
```

## Kullanım

### Görüntü Analizi

- JPG/JPEG/PNG formatında görüntü yükleyin.
- "Analiz Et" butonuna basın.

### Canlı Yayın

- IP Webcam URL'si girin.
- Örnek: http://IP_ADRESI:8080/video
- "Başlat" butonuna basın.

### Video Analizi

- MP4 video dosyası yükleyin.
- "Videoyu Analiz Et" butonuna basın.

## Model

Uygulama, proje kök dizininde bulunan `best.pt` model dosyasını kullanır. Dosya yoksa sistem pasif görünür ve tespitler çalışmaz.

## Risk Skoru Mantığı

Başlangıç skoru 100'dür. Kasksız tespitinde skor 40 puan azalır. Minimum skor 0'dır.

## Teknolojiler

- Python
- Streamlit
- Ultralytics YOLO
- OpenCV
- Pillow
- NumPy
- Plotly

## Proje Hakkında

- Üniversite dönem projesi olarak geliştirilmiştir.
- Kurye güvenliği ve iş güvenliği senaryoları için tasarlanmıştır.

## Emeği Geçenler
-Semih Kadir Yıldırım

-Ayşe Pelin Demirel 
