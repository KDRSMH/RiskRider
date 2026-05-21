# RiskRider

RiskRider, motosiklet sürücüsünün fotoğrafını yükleyerek çok faktörlü risk analizi yapan ve 0-100 arası Risk Skoru üreten yapay zeka destekli bir web uygulamasıdır. Sistem, YOLOv11 tespitleri ve Streamlit arayüzü ile sürücü güvenliğini otomatik olarak değerlendirir.

## Projenin Mevcut Durumu

- YOLOv11 (pretrained `yolo11n.pt`) ile `person` ve `motorcycle` tespiti yapar
- Streamlit tabanlı web arayüzü çalışır
- Risk skoru hesaplama motoru hazırdır (0-100 arası)
- Çok faktörlü risk analizi altyapısı kuruludur
- Bounding box ile görsel tespit üretir
- Güvenlik metrikleri ve bireysel tespit listesi gösterilir

## Özellikler

- Görüntü yükleme ve tek tıkla analiz
- Tespitli görüntü üretimi ve görselleştirme
- Risk skoru ve seviye etiketi
- Tetiklenen risk faktörleri listesi
- Güvenlik metrikleri (kask/yelek/telefon vb.)
- Modern ve sade Streamlit arayüzü

## Proje Yapısı

```
riskrider/
├── app.py
├── detect.py
├── requirements.txt
└── README.md
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```bash
python -m streamlit run app.py
```

Ardından tarayıcınızda açılan arayüzden bir sürücü fotoğrafı yükleyin ve "Analiz Et" butonuna tıklayın.

## Risk Skoru Mantığı

Başlangıç skoru 100'dür. Aşağıdaki risk faktörleri tespit edilirse skor düşer:

- Kasksız: -40
- Telefon kullanımı: -25
- Yeleksiz: -20
- Aşırı yük: -10
- Ek yolcu: -5

Minimum skor 0'dır.

## Teknolojiler

- Python
- YOLOv11 (Ultralytics)
- OpenCV
- Streamlit
- Pillow
- NumPy

## 🔜 Planlanan Özellikler

1. Özel Eğitilmiş Model (best.pt)
	- Roboflow veri seti ile eğitilecek
	- Tespit edilecek sınıflar: helmet, no_helmet, vest, no_vest, phone_use, overloaded, passenger
	- Google Colab üzerinde YOLOv11 fine-tuning

2. IP Kamera / RTSP Stream Desteği
	- Telefon IP Webcam uygulaması ile mobese simülasyonu
	- Gerçek zamanlı risk analizi
	- Stream URL girişi ile bağlantı

3. Video Dosyası Analizi
	- MP4 video yükleme
	- Kare kare otomatik analiz
	- Sonuç videosu indirme
	- Zaman bazlı risk grafiği

## Notlar

- Mevcut model pretrained olduğu için kask/yelek sınıflarını henüz tanımıyor
- Risk skoru altyapısı hazır; eğitilmiş model bağlandığında tam çalışacak
- Final sunumunda tüm planlanan özellikler aktif olacak

## Proje Hakkında

- Üniversite dönem projesi olarak geliştirilmiştir
- Yemeksepeti, Trendyol Go, Getir gibi kurye firmalarının sürücü güvenliğini otomatik denetlemesi senaryosu üzerine tasarlanmıştır
- Trafik güvenliği ve iş güvenliği alanında yapay zeka uygulamasıdır

## Gereksinimler

```
streamlit
opencv-python
ultralytics
pillow
numpy
```
