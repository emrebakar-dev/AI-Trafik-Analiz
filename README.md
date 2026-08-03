# KGM Trafik Yoğunluğu İzleme Projesi

Bu proje, bir video dosyası üzerinde araç tespiti ve sayımı yaparak trafik yoğunluğunu ölçer. Ultralytics YOLO modelini kullanır ve analiz sonuçlarını hem ekranda gösterebilir hem de bir `.txt` raporu olarak kaydeder.

## Özellikler

- Video üzerinden araç algılama ve takip
- Geçiş hattından geçen araç sayısını hesaplama
- 10 saniyelik pencere içinde araç yoğunluğunu `Dusuk`, `Orta` veya `Yuksek` olarak sınıflandırma
- Analiz ilerlemesi ve sayı bilgilerinin video üzerine yazdırılması
- Analiz sonu raporunun `reports/` klasörüne kaydedilmesi

## Dosya Yapısı

- `main.py`: Projenin ana analiz ve raporlama kodu
- `requirements.txt`: Projenin Python bağımlılıkları
- `trafik.mov`: Analiz edilen örnek video dosyası
- `yolo11s.pt`, `yolo11n.pt`: Kullanılabilecek YOLO model dosyaları
- `reports/`: Analiz tamamlandıktan sonra oluşturulan raporların saklandığı klasör
- `assets/`: Program çalışırken alınan ekran görüntüsü ve ek destek dosyaları

## Gereksinimler

- Python 3.11 veya daha yeni bir sürüm önerilir
- `requirements.txt` içindeki paketler

### Kurulum

1. Sanal ortam oluşturun ve etkinleştirin:

```bash
python -m venv venv
source venv/bin/activate
```

2. Gereksinimleri yükleyin:

```bash
pip install -r requirements.txt
```

## Kullanım

1. `main.py` içinde `VIDEO_PATH` ve `MODEL_NAME` ayarlarının doğru olduğundan emin olun.
2. Proje dizinindeyken script'i çalıştırın:

```bash
python main.py
```

3. Video işlenirken pencere açılır ve analiz bilgileri ekranda gösterilir.
4. Analizi durdurmak için `q` tuşuna basın.

## Raporlama

Analiz tamamlandığında veya kullanıcı erken bitirdiğinde, sonuçlar otomatik olarak `reports/` dizinine bir `.txt` dosyası olarak kaydedilir. Rapor içinde şunlar yer alır:

- Analiz tarihi
- Video dosyası adı
- Toplam video süresi
- Analiz edilen süre
- Toplam sayılan araç sayısı
- En yoğun 10 saniyedeki araç sayısı
- En yüksek trafik yoğunluğu
- Son trafik yoğunluğu

## Ekran Görüntüsü

`assets/` klasöründe programın çalışırken alınmış bir ekran görüntüsü bulunur. Bu ekran görüntüsü, modelin tespit ve analiz sonuçlarını nasıl gösterdiğini anlamanıza yardımcı olur.

- `assets/Screenshot 2026-08-03 at 10.24.19.png`: Analiz sırasında gösterilen video ve metrikler

## Ayarlar

Aşağıdaki değerler `main.py` içinde kolayca değiştirilebilir:

- `CONFIDENCE_THRESHOLD`: Algılama güven eşiği
- `IMAGE_SIZE`: Model için kullanılan giriş boyutu
- `DENSITY_WINDOW_SECONDS`: Yoğunluk hesaplama penceresi
- `LOW_DENSITY_LIMIT`, `MEDIUM_DENSITY_LIMIT`: Yoğunluk sınıflandırma eşikleri
- `VEHICLE_CLASS_IDS`: Tespit edilecek araç sınıfları

## Notlar

Bu proje staj kapsamında geliştirilmiş bir prototiptir. Sonuçlar gerçek trafik koşullarından farklılık gösterebilir.
