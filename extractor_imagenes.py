import os
import cv2

def extract_frames(video_path, frames_per_second=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frames_per_second == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames

def extract_frames_from_folder(folder_path, frames_per_second=1):
    video_extensions = [".mp4", ".avi", ".mkv", ".mov"]  # Lista de extensiones de archivos de video

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and file.lower().endswith(tuple(video_extensions)):
            print(f"Extrayendo imágenes del video: {file}")
            frames = extract_frames(file_path, frames_per_second)

            # Guardar las imágenes en el disco
            for i, frame in enumerate(frames):
                file_name = os.path.splitext(file)[0]
                cv2.imwrite(f"{file_name}_frame_{i}.jpg", frame)

if __name__ == "__main__":
    folder_path = "content/robos"  # Reemplaza con la ruta de la carpeta que contiene los videos
    frames_per_second = 5  # Número de cuadros por segundo a extraer

    extract_frames_from_folder(folder_path, frames_per_second)
