import whisper
import pyaudio
import numpy as np

# 加载中等大小的 Whisper 模型
model = whisper.load_model("medium")

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

audio_interface = pyaudio.PyAudio()

stream = audio_interface.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              frames_per_buffer=CHUNK)

print("🎙️ 开始录音（按 Ctrl+C 停止）")

try:
    while True:
        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(np.frombuffer(data, np.int16))

        audio_np = np.hstack(frames).astype(np.float32) / 32768.0

        # 识别音频内容，设置为英语
        result = model.transcribe(audio_np, language='en')
        print("📝 识别结果:", result['text'])

except KeyboardInterrupt:
    print("\n⏹️ 停止录音")
    stream.stop_stream()
    stream.close()
    audio_interface.terminate()
