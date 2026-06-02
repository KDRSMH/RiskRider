# RiskRider — Motosiklet Sürücü Risk Analiz Sistemi
## Proje Final Raporu

**Proje Adı:** RiskRider  
**Konu:** Yapay Zeka Destekli Motosiklet Sürücüsü Güvenlik Analizi  
**Kullanılan Teknolojiler:** YOLOv11, Python, Streamlit, OpenCV, Roboflow  
**Rapor Tarihi:** Haziran 2026  

---

## İçindekiler

1. [Giriş ve Proje Amacı](#1-giriş-ve-proje-amacı)
2. [Sistem Mimarisi](#2-sistem-mimarisi)
3. [Veri Toplama](#3-veri-toplama)
4. [Veri Etiketleme](#4-veri-etiketleme)
5. [Veri Bölme](#5-veri-bölme)
6. [Model Eğitimi](#6-model-eğitimi)
7. [Model Değerlendirme](#7-model-değerlendirme)
8. [Çıkarım ve Test](#8-çıkarım-ve-test)
9. [Web Arayüzü](#9-web-arayüzü)
10. [Kullanıcı Etkileşimi](#10-kullanıcı-etkileşimi)
11. [Sonuçların Görselleştirilmesi](#11-sonuçların-görselleştirilmesi)
12. [Sonuç ve Değerlendirme](#12-sonuç-ve-değerlendirme)

---

## 1. Giriş ve Proje Amacı

Türkiye'de ve dünyada motosiklet kaynaklı trafik kazaları, sürücülerin kask ve koruyucu yelek gibi güvenlik ekipmanlarını kullanmaması nedeniyle büyük ölçüde can kaybına yol açmaktadır. Yemeksepeti, Trendyol Go ve Getir gibi kurye şirketlerinin yaygınlaşmasıyla birlikte yoğun trafikte motosiklet kullanımı artmış; sürücü güvenliğinin otomatik ve ölçeklenebilir biçimde denetlenmesi ihtiyacı doğmuştur.

**RiskRider**, bu ihtiyaca yanıt vermek üzere geliştirilmiş, yapay zeka destekli bir motosiklet sürücüsü güvenlik analiz sistemidir. Sistem, bir fotoğraf, video veya canlı kamera görüntüsü üzerinden sürücünün güvenlik ihlallerini otomatik olarak tespit etmekte ve 0–100 arasında dinamik bir **Risk Skoru** üretmektedir.

### Projenin Temel Hedefleri

- Motosiklet sürücülerinin kask kullanıp kullanmadığını gerçek zamanlı olarak tespit etmek
- Tespit edilen ihlallere göre nesnel ve sayısal bir risk skoru hesaplamak
- Kurumsal kullanıma uygun, modern ve kullanıcı dostu bir web arayüzü sunmak
- Fotoğraf, video ve canlı RTSP/HTTP stream üzerinde çalışabilir, çok modlu bir analiz platformu oluşturmak

---

## 2. Sistem Mimarisi

RiskRider üç temel bileşenden oluşmaktadır:

```
┌─────────────────────────────────────────────────┐
│                 KULLANICI ARAYÜZÜ               │
│         Streamlit Dashboard (app.py)            │
│   [Fotoğraf Sekmesi | Kamera Sekmesi | Video]   │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│              TESPİT MOTORU                      │
│           detect.py / stream.py                 │
│    YOLOv11 (best.pt) + OpenCV + Pillow          │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│             RİSK SKORU MOTORU                   │
│         calculate_risk_score()                  │
│  Sınıf bazlı ağırlıklı puan düşme algoritması  │
└─────────────────────────────────────────────────┘
```

| Bileşen | Dosya | Görev |
|---|---|---|
| Arayüz | `app.py` | Streamlit tabanlı 3 sekmeli dashboard |
| Tespit Motoru | `detect.py` | YOLO çıkarımı, bounding box, sınıf eşleme |
| Stream Modülü | `stream.py` | RTSP/HTTP canlı akış yönetimi |
| Video Modülü | `video.py` | Kare kare video analizi ve grafik üretimi |
| Model | `best.pt` | Fine-tuned YOLOv11-nano ağırlıkları |

---

## 3. Veri Toplama

Projenin makine öğrenmesi bileşeni için gerekli eğitim verisi, açık kaynak veri seti platformu olan **Roboflow Universe** üzerinden temin edilmiştir.

### Veri Seti Özellikleri

| Özellik | Değer |
|---|---|
| Kaynak Platform | Roboflow Universe |
| Veri Seti Odağı | Motosiklet ve kask güvenliği |
| Toplam Görsel Sayısı | ~1.200 görsel |
| Lisans | CC BY 4.0 |
| Görüntü Formatı | JPG / PNG |

Proje gereksinimlerinde belirlenen **minimum 200 görsel** eşiği, toplanan ~1.200 görsel ile yaklaşık **6 kat** aşılmıştır. Bu durum, modelin daha geniş bir veri dağılımı üzerinde eğitilmesine ve daha genelleştirilebilir ağırlıklar elde etmesine olanak tanımıştır.

Veri seti, farklı ışık koşulları, kamera açıları, sürücü pozisyonları ve motosiklet türlerini kapsayan çeşitli görüntüler içermekte olup gerçek dünya senaryolarına yakın bir dağılım sergilemektedir.

---

## 4. Veri Etiketleme

Veri etiketleme süreci, Roboflow'un yerleşik annotation arayüzü kullanılarak gerçekleştirilmiştir. Tüm görseller **nesne tespiti (Object Detection)** görevine uygun şekilde, sınırlayıcı kutular (bounding boxes) ile etiketlenmiştir.

### Tanımlanan Sınıflar

| Sınıf ID | Sınıf Adı | Türkçe Karşılığı | Açıklama |
|---|---|---|---|
| 0 | `A_helmet_not_worn` | Kasksız | Kask takmayan sürücü veya yolcu |
| 1 | `B_helmet_worn` | Kasklı | Kask takan sürücü veya yolcu |
| 2 | `C_Motorcycle` | Motosiklet | Motosiklet aracının kendisi |
| 3 | `D_person` | Kişi | Motosiklet üzerindeki veya yakınındaki kişi |

Sınıf isimlendirmesinde alfabetik ön ek (`A_`, `B_`, `C_`, `D_`) kullanılmış; bu yaklaşım sınıfların model çıktısında tutarlı bir sırayla listelenmesini ve `CLASS_NAMES` sözlüğü üzerinden kolayca Türkçe karşılıklarına eşlenmesini sağlamıştır.

---

## 5. Veri Bölme

Eğitim sürecinin sağlıklı yürütülmesi ve modelin görülmemiş veriler üzerindeki performansının nesnel biçimde ölçülebilmesi için veri seti Roboflow tarafından otomatik olarak iki alt kümeye ayrılmıştır.

### Bölme Oranları

| Alt Küme | Oran | Görsel Sayısı | Kullanım Amacı |
|---|---|---|---|
| **Eğitim (Train)** | %80 | 933 | Model ağırlıklarının güncellenmesi |
| **Doğrulama (Validation)** | %20 | 267 | Her epoch sonunda performans ölçümü |

%80 / %20 oranı, makine öğrenmesi literatüründe nesne tespiti görevleri için yaygın kabul görmüş standart bir bölme stratejisidir. Doğrulama kümesinin eğitim sürecinde hiçbir zaman model tarafından görülmemesi, elde edilen metriklerin güvenilirliğini garanti altına almaktadır.

---

## 6. Model Eğitimi

### Eğitim Ortamı

| Parametre | Değer |
|---|---|
| Platform | Google Colab |
| GPU | NVIDIA Tesla T4 (16 GB VRAM) |
| İşletim Sistemi | Linux (Ubuntu) |
| Python Sürümü | 3.12 |
| Kütüphane | Ultralytics YOLOv11 |

### Model Seçimi

Temel model olarak **YOLOv11-nano (`yolo11n.pt`)** tercih edilmiştir. Nano varyantının seçilmesinin başlıca gerekçeleri şunlardır:

- **Hız:** Gerçek zamanlı çıkarım gereksinimini (canlı kamera akışı) karşılayabilecek düşük gecikme süresi
- **Kaynak verimliliği:** Lokal ortamda ve sınırlı donanımda çalışabilirlik
- **Fine-tuning uygunluğu:** Az sayıda parametre, küçük/orta ölçekli veri setlerinde aşırı öğrenmeye (overfitting) karşı daha dirençlidir

### Hiperparametreler

| Hiperparametre | Değer | Açıklama |
|---|---|---|
| `epochs` | 50 | Toplam eğitim turu sayısı |
| `imgsz` | 640 | Giriş görüntüsü boyutu (piksel) |
| `batch` | 16 | Her adımda işlenen görüntü sayısı |
| `device` | cuda (T4) | Eğitim donanımı |
| `patience` | 20 | Early stopping eşiği |
| `base_model` | yolo11n.pt | COCO pretrained başlangıç ağırlıkları |

### Eğitim Süreci

Model, COCO veri seti üzerinde önceden eğitilmiş (`pretrained`) YOLOv11-nano ağırlıklarından başlayarak **transfer learning** yaklaşımıyla fine-tuning işlemine tabi tutulmuştur. 50 epoch boyunca eğitim kaybı (training loss) ve doğrulama metrikleri izlenmiş; en yüksek doğrulama mAP50 değerini veren epoch'un ağırlıkları otomatik olarak **`best.pt`** olarak kaydedilmiştir.

---

## 7. Model Değerlendirme

50. epoch tamamlandıktan sonra model, eğitim sürecinde hiç görmediği **267 doğrulama görseli** üzerinde test edilmiştir. Aşağıdaki tablo, bu değerlendirme sürecinde elde edilen metrikleri sınıf bazında özetlemektedir.

### Doğrulama Sonuçları

| Sınıf | Görüntü | Örnek | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **Tüm Sınıflar (all)** | **267** | **893** | **0.858** | **0.799** | **0.860** | **0.542** |
| A_helmet_not_worn (Kasksız) | — | 106 | 0.829 | 0.825 | 0.872 | 0.539 |
| B_helmet_worn (Kasklı) | — | 246 | 0.931 | 0.882 | **0.946** | 0.703 |
| C_Motorcycle (Motosiklet) | — | 199 | 0.849 | 0.732 | 0.806 | 0.504 |
| D_person (Kişi) | — | 342 | 0.825 | 0.757 | 0.816 | 0.423 |

### Metriklerin Yorumu

**Genel Performans**

Model, tüm sınıflar genelinde **%85.8 Precision** ve **%79.9 Recall** değerlerine ulaşmıştır. Ortalama mAP50 değeri **0.860** olarak ölçülmüş olup bu sonuç, gerçek dünya nesne tespiti uygulamaları için kabul edilebilir eşiğin (0.50) oldukça üzerindedir.

**Öne Çıkan Sınıf: B_helmet_worn (Kasklı)**

Projenin odak sınıfı olan **Kasklı (B_helmet_worn)** tespitinde model **%94.6 mAP50** ile en yüksek başarıyı sergilemiştir. Bu değer, modelin kask takan bir sürücüyü neredeyse hatasız biçimde tespit edebildiğini göstermektedir. Precision değerinin **0.931** olması, yanlış pozitif (false positive) oranının son derece düşük tutulduğuna işaret etmektedir.

**Kasksız Tespit (A_helmet_not_worn)**

Güvenlik ihlalinin doğrudan göstergesi olan Kasksız sınıfı **%87.2 mAP50** ile güçlü bir performans ortaya koymuştur. Recall değerinin **0.825** olması, gerçek ihlallerin %82.5'inin sistemden kaçmadan tespit edilebildiği anlamına gelmektedir.

**Motosiklet ve Kişi Sınıfları**

Motosiklet (%80.6 mAP50) ve Kişi (%81.6 mAP50) sınıflarındaki görece düşük performans, bu nesnelerin kalabalık sahnelerde birbirleriyle örtüşmesinden ve boyut varyasyonundan kaynaklanmaktadır. Ek veri artırma teknikleri veya daha büyük model varyantı (yolo11s/m) ile bu değerlerin iyileştirilebileceği öngörülmektedir.

---

## 8. Çıkarım ve Test

Eğitim tamamlandıktan sonra elde edilen **`best.pt`** ağırlık dosyası Google Drive üzerinden lokal ortama aktarılmış ve `detect.py` modülünde `MODEL_PATH = "best.pt"` olarak yapılandırılmıştır.

### Test Senaryoları

**Statik Görüntü Testi**

Model, eğitim ve doğrulama veri setlerinde yer almayan, daha önce hiç görmediği motosiklet fotoğrafları üzerinde test edilmiştir. Ultralytics `YOLO.predict()` metoduyla gerçekleştirilen çıkarımda model, kask durumunu, motosikleti ve kişileri başarıyla sınıflandırmış; her tespit için güven skoru (%confidence) ve sınırlayıcı kutu koordinatları üretmiştir.

**Canlı Video Akışı Testi**

Android tabanlı **IP Webcam** uygulaması aracılığıyla yerel ağ üzerinden sağlanan RTSP/HTTP video akışı `stream.py` modülü vasıtasıyla OpenCV'nin `cv2.VideoCapture()` fonksiyonuna bağlanmıştır. Her N. kare (throttling ile) işlenerek model çıkarımı gerçek zamanlı olarak yürütülmüş ve sistem canlı akışta sınıfları başarıyla tespit etmiştir.

### Tespit Sonrası İşleme

Ultralytics'in `.plot()` metodu ile üretilen BGR formatındaki annotated görüntüler, RGB'ye dönüştürülerek (`[:, :, ::-1]`) Pillow `Image` nesnesi haline getirilmekte ve Streamlit arayüzüne aktarılmaktadır. Bu süreç `detect.py`'deki `run_detection()` fonksiyonu tarafından yönetilmektedir.

---

## 9. Web Arayüzü

### Teknoloji Seçimi

Uygulama arayüzü **Streamlit** kütüphanesi ile geliştirilmiştir. Python ekosistemiyle tam entegrasyon, hızlı prototipleme ve `st.session_state` ile durum yönetimi Streamlit'in tercih edilmesinin başlıca nedenleridir.

### Arayüz Özellikleri

**Tema ve Tasarım**

Arayüz, `#0a0f1e` arka plan rengiyle tam karanlık tema (dark mode) üzerine inşa edilmiştir. Standart Streamlit bileşenleri özel CSS enjeksiyonu (`st.markdown(CSS, unsafe_allow_html=True)`) ile yeniden stillendirilmiş; header, toolbar ve varsayılan Streamlit markalama gizlenmiştir.

```
Renk Paleti:
─ Arka Plan:  #0a0f1e  (koyu lacivert)
─ Kart:       #111827  (koyu gri)
─ Vurgu:      #3b82f6  (mavi)
─ Başarı:     #10b981  (yeşil)
─ Uyarı:      #f59e0b  (sarı)
─ Tehlike:    #ef4444  (kırmızı)
```

**Sekme Yapısı**

Uygulama üç ana sekmeden oluşmaktadır:

| Sekme | İkon | İçerik |
|---|---|---|
| Fotoğraf Analizi | 📷 | Görüntü yükleme ve tek seferlik analiz |
| Canlı Stream | 📡 | RTSP/HTTP URL ile gerçek zamanlı analiz |
| Video Analizi | 🎬 | MP4 yükleme, kare kare işleme, grafik |

**Sidebar**

Sol panel üzerinden güven skoru eşiği (`confidence`) 0.10–0.90 aralığında `st.slider` ile ayarlanabilmektedir. Ayrıca model durumu (Sistem Aktif / Pasif) göstergesi yer almaktadır.

---

## 10. Kullanıcı Etkileşimi

### Fotoğraf Yükleme

`st.file_uploader` bileşeni ile kullanıcılar **JPG, JPEG ve PNG** formatlarında görüntü yükleyebilmektedir. Yüklenen görüntü Pillow `Image.open()` ile RGB formatına dönüştürüldükten sonra `run_detection()` fonksiyonuna iletilmektedir.

```python
upload = st.file_uploader("Görüntü yükleyin", type=["jpg", "jpeg", "png"])
original_image = Image.open(io.BytesIO(upload.read())).convert("RGB")
```

### Canlı Stream Bağlantısı

Kullanıcı, `st.text_input` alanına RTSP veya HTTP MJPEG stream URL'si girerek canlı analizi başlatabilmektedir. Desteklenen URL formatları:

- `rtsp://192.168.x.x:8080/h264_ulaw.sdp` (RTSP protokolü)
- `http://192.168.x.x:8080/video` (HTTP MJPEG, IP Webcam uygulaması)

### Video Yükleme

MP4, AVI veya MOV formatındaki video dosyaları `st.file_uploader` ile yüklenebilmektedir. Yüklenen dosya geçici diske yazılarak `analyze_video()` fonksiyonu tarafından kare kare işlenmektedir.

---

## 11. Sonuçların Görselleştirilmesi

### Sınırlayıcı Kutular (Bounding Boxes)

Tespit edilen her nesne, sınıfa özgü renk kodlamasıyla ekrana yansıtılmaktadır:

| Sınıf | Renk | RGB Değeri |
|---|---|---|
| Kasksız | Kırmızı | `(220, 20, 60)` |
| Kasklı | Yeşil | `(46, 204, 113)` |
| Motosiklet | Mavi | `(52, 152, 219)` |
| Kişi | Gri | `(130, 130, 130)` |

### Risk Skoru Algoritması

Risk skoru, **100 puan** başlangıç değerinden sınıf bazlı cezaların düşülmesiyle hesaplanmaktadır:

**Risk Seviyeleri:**

| Skor Aralığı | Seviye |
|---|---|
| 80 – 100 | 🟢 Düşük Risk |
| 50 – 79 | 🟡 Orta Risk |
| 20 – 49 | 🟠 Yüksek Risk |
| 0 – 19 | 🔴 Kritik Risk |

### Güvenlik Metrikleri Paneli

Analiz sonrasında beş kategori için sayaç tabanlı metrik grid'i gösterilmektedir: Kasklı, Kasksız, Yelekit, Yeleksiz, Telefon Kullanan. Her metrik kartı, ilgili renkle vurgulanmış bir çizgiyle kategori bazlı görsel hiyerarşi sağlamaktadır.

### Video Analizi Grafiği

Video analizi tamamlandığında Plotly kütüphanesi ile zaman bazlı risk skoru grafiği üretilmekte; Düşük/Orta/Yüksek Risk eşik çizgileri referans olarak görselleştirilmektedir. Ek olarak annotated video dosyası `st.download_button` ile indirilebilmektedir.

---

## 12. Sonuç ve Değerlendirme

RiskRider projesi, belirlenen tüm proje gereksinimlerini başarıyla karşılamıştır. Proje kapsamında gerçekleştirilen çalışmalar şu başlıklar altında özetlenebilir:

**Teknik Başarılar**

- Roboflow Universe üzerinden ~1.200 görsel içeren, minimum eşiğin 6 katı büyüklüğünde bir veri seti derlenerek etiketlenmiştir.
- YOLOv11-nano modeli Google Colab ortamında Tesla T4 GPU ile fine-tuning yapılmış; **%86.0 mAP50** genel başarı oranına ulaşılmıştır.
- Projenin kritik sınıfı olan Kasklı tespitinde **%94.6 mAP50** ile yüksek doğruluk elde edilmiştir.
- Eğitilen model, statik görüntülerde ve IP Webcam üzerinden gelen canlı akışlarda başarıyla çalıştırılmıştır.

**Başarılar**

- Modüler mimari: `detect.py`, `stream.py`, `video.py` bağımsız modüller olarak tasarlanmıştır.
- Özel CSS ve dark mode tema ile profesyonel Streamlit dashboard geliştirilmiştir.
- Fotoğraf, canlı stream ve video analizi için 3 sekmeli çok modlu arayüz hayata geçirilmiştir.
- Risk skoru algoritması sınıf bazlı ağırlıklı puan düşme sistemiyle gerçekçi bir güvenlik değerlendirmesi sunmaktadır.

---

*Bu rapor, RiskRider v1.0 BETA sürümünü kapsamaktadır.*
