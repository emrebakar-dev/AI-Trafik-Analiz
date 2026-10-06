from collections import deque
from datetime import datetime
from pathlib import Path

print("Kutuphaneler yukleniyor (ilk acilis biraz surebilir)...", flush=True)

import cv2
import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent

REPORTS_DIR = BASE_DIR / "reports"
VIDEO_PATH = BASE_DIR / "trafik.mov"
MODEL_NAME = str(BASE_DIR / "yolo11s.pt")

CONFIDENCE_THRESHOLD = 0.35
IMAGE_SIZE = 640

MAX_FRAME_WIDTH = 1280
FRAME_STRIDE = 2


def select_device() -> str:
    """Mevcut en hizli cihazi secer (CUDA > Apple MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


DEVICE = select_device()

LINE_MARGIN = 20

DENSITY_WINDOW_SECONDS = 10

LOW_DENSITY_LIMIT = 3
MEDIUM_DENSITY_LIMIT = 7

VEHICLE_CLASS_IDS = [2, 3, 5, 7]


def calculate_density(vehicle_count: int) -> str:
    """Son zaman araligindaki arac sayisina gore yogunluk hesaplar."""
    if vehicle_count <= LOW_DENSITY_LIMIT:
        return "Dusuk"

    if vehicle_count <= MEDIUM_DENSITY_LIMIT:
        return "Orta"

    return "Yuksek"


def format_time(seconds: float) -> str:
    """Saniyeyi dakika:saniye bicimine cevirir."""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    return f"{minutes:02}:{remaining_seconds:02}"


def create_txt_report(
    video_name: str,
    analyzed_duration: float,
    total_video_duration: float,
    total_count: int,
    maximum_recent_count: int,
    maximum_density: str,
    current_density: str,
    analysis_completed: bool,
) -> Path:
    """Analiz sonucunu TXT raporu olarak kaydeder."""
    REPORTS_DIR.mkdir(exist_ok=True)

    analysis_date = datetime.now()

    report_name = (
        "trafik_analiz_raporu_"
        f"{analysis_date.strftime('%Y%m%d_%H%M%S')}.txt"
    )

    report_path = REPORTS_DIR / report_name

    completion_status = (
        "Tamamlandi"
        if analysis_completed
        else "Kullanici tarafindan erken sonlandirildi"
    )

    report_content = f"""KGM AKILLI TRAFIK ANALIZ RAPORU
=======================================================

Analiz tarihi          : {analysis_date.strftime('%d.%m.%Y %H:%M:%S')}
Video dosyasi          : {video_name}
Toplam video suresi    : {total_video_duration:.1f} saniye
Analiz edilen sure     : {analyzed_duration:.1f} saniye
Analiz durumu          : {completion_status}
Kullanilan model       : {MODEL_NAME}
Guven esigi            : %{CONFIDENCE_THRESHOLD * 100:.0f}
Yogunluk penceresi     : {DENSITY_WINDOW_SECONDS} saniye

ANALIZ SONUCLARI
-------------------------------------------------------
Toplam sayilan arac    : {total_count}
En yogun 10 saniye     : {maximum_recent_count} arac
En yuksek yogunluk     : {maximum_density}
Son yogunluk durumu    : {current_density}

YOGUNLUK SINIRLARI
-------------------------------------------------------
0 - {LOW_DENSITY_LIMIT} arac            : Dusuk
{LOW_DENSITY_LIMIT + 1} - {MEDIUM_DENSITY_LIMIT} arac            : Orta
{MEDIUM_DENSITY_LIMIT + 1} ve uzeri arac      : Yuksek

NOTLAR
-------------------------------------------------------
Bu proje staj kapsaminda gelistirilmis bir prototiptir.

Sonuclar, gercek sonuclardan farkli olabilir.
"""

    report_path.write_text(report_content, encoding="utf-8")

    return report_path


def main() -> None:
    if not VIDEO_PATH.exists():
        print(f"Video bulunamadi: {VIDEO_PATH}")
        return

    SF_DATALESS = 0x40000000
    if getattr(VIDEO_PATH.stat(), "st_flags", 0) & SF_DATALESS:
        print(
            "UYARI: trafik.mov iCloud'da duruyor, bilgisayara indirilmemis.\n"
            "Finder'da dosyaya sag tiklayip 'Simdi Indir' secin veya projeyi\n"
            "Masaustu/iCloud disindaki bir klasore tasiyin, sonra tekrar deneyin."
        )
        return

    print(f"Model yukleniyor: {Path(MODEL_NAME).name} ({DEVICE})", flush=True)
    model = YOLO(MODEL_NAME)
    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        print("Video acilamadi.")
        return

    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames > 0:
        total_video_duration = total_frames / fps
    else:
        total_video_duration = 0.0

    counted_ids = set()
    previous_positions = {}

    crossing_times = deque()

    total_count = 0
    frame_number = 0

    maximum_recent_count = 0
    maximum_density = "Dusuk"
    current_density = "Dusuk"

    analysis_completed = True

    print("Analiz basliyor. Cikmak icin 'q' tusuna basin.", flush=True)

    while True:
        skipped = True
        for _ in range(FRAME_STRIDE - 1):
            if not video.grab():
                skipped = False
                break
            frame_number += 1

        if not skipped:
            break

        success, frame = video.read()

        if not success:
            break

        frame_number += 1
        current_time = frame_number / fps

        frame_height, frame_width = frame.shape[:2]

        if frame_width > MAX_FRAME_WIDTH:
            scale = MAX_FRAME_WIDTH / frame_width
            frame = cv2.resize(
                frame,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
            frame_height, frame_width = frame.shape[:2]

        line_y = frame_height // 2

        line_start_x = int(frame_width * 0.32)
        line_end_x = frame_width

        cv2.line(
            frame,
            (line_start_x, line_y),
            (line_end_x, line_y),
            (0, 0, 255),
            3,
        )

        results = model.track(
            frame,
            persist=True,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            half=(DEVICE == "cuda"),
            classes=VEHICLE_CLASS_IDS,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        for result in results:
            for box in result.boxes:
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                inside_counting_area = (
                    line_start_x <= center_x <= line_end_x
                )

                if not inside_counting_area:
                    continue

                previous_y = previous_positions.get(track_id)

                if track_id not in counted_ids:
                    crossed_line = False

                    if previous_y is not None:
                        crossed_line = (
                            previous_y < line_y <= center_y
                            or previous_y > line_y >= center_y
                        )

                    elif abs(center_y - line_y) <= LINE_MARGIN:
                        crossed_line = True

                    if crossed_line:
                        counted_ids.add(track_id)
                        total_count += 1
                        crossing_times.append(current_time)

                previous_positions[track_id] = center_y

                label = (
                    f"Arac "
                    f"ID:{track_id} "
                    f"%{confidence * 100:.0f}"
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (255, 0, 0),
                    -1,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        while (
            crossing_times
            and current_time - crossing_times[0]
            > DENSITY_WINDOW_SECONDS
        ):
            crossing_times.popleft()

        recent_vehicle_count = len(crossing_times)
        current_density = calculate_density(recent_vehicle_count)

        if recent_vehicle_count > maximum_recent_count:
            maximum_recent_count = recent_vehicle_count
            maximum_density = current_density

        if total_video_duration > 0:
            progress = min(
                (current_time / total_video_duration) * 100,
                100,
            )
        else:
            progress = 0.0

        cv2.putText(
            frame,
            f"Toplam Arac: {total_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Son 10 sn Arac: {recent_vehicle_count}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Trafik Yogunlugu: {current_density}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            (
                f"Sure: {format_time(current_time)} / "
                f"{format_time(total_video_duration)}"
            ),
            (20, frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        progress_text = f"Analiz: %{progress:.0f}"

        text_width = cv2.getTextSize(
            progress_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2,
        )[0][0]

        cv2.putText(
            frame,
            progress_text,
            (frame_width - text_width - 20, frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "KGM Akilli Trafik Analiz Sistemi",
            frame,
        )

        pressed_key = cv2.waitKey(1) & 0xFF

        if pressed_key == ord("q"):
            analysis_completed = False
            break

    analyzed_duration = frame_number / fps

    video.release()
    cv2.destroyAllWindows()

    report_path = create_txt_report(
        video_name=VIDEO_PATH.name,
        analyzed_duration=analyzed_duration,
        total_video_duration=total_video_duration,
        total_count=total_count,
        maximum_recent_count=maximum_recent_count,
        maximum_density=maximum_density,
        current_density=current_density,
        analysis_completed=analysis_completed,
    )

    status_text = (
        "Analiz tamamlandi."
        if analysis_completed
        else "Analiz erken sonlandirildi."
    )

    print("\n" + "=" * 55)
    print("KGM TRAFIK ANALIZ SONUCU")
    print("=" * 55)
    print(f"Durum: {status_text}")
    print(f"Video: {VIDEO_PATH.name}")
    print(f"Toplam video suresi: {total_video_duration:.1f} saniye")
    print(f"Analiz edilen sure: {analyzed_duration:.1f} saniye")
    print(f"Toplam sayilan arac: {total_count}")
    print(
        f"En yogun {DENSITY_WINDOW_SECONDS} saniyedeki arac: "
        f"{maximum_recent_count}"
    )
    print(f"En yuksek trafik yogunlugu: {maximum_density}")
    print(f"Son trafik yogunlugu: {current_density}")
    print(f"Rapor kaydedildi: {report_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()