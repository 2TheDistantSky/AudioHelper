import torch
import torchaudio
import torchaudio.transforms as T
from scipy.spatial.distance import cosine
from tqdm import tqdm


# === 配置区域 ===
AUDIO_FILE = "./your_audio.mp3"
START_TIME_STR = "0:0:0"
END_TIME_STR = "0:1:43"
STEP_SIZE = 0.1  # 滑动窗口步长，单位：秒
THRESHOLD = 0.06  # 相似度阈值（越小越严格）
INTERVAL = 10  # 秒


# === 时间解析函数 ===
def parse_time_string(time_str: str) -> int:
    """将 hh:mm:ss / mm:ss / ss 字符串转为总秒数"""
    parts = [int(p) for p in time_str.strip().split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0
        m, s = parts
    elif len(parts) == 1:
        h = 0
        m = 0
        s = parts[0]
    else:
        raise ValueError(f"时间格式错误: {time_str}")
    return h * 3600 + m * 60 + s


# === 时间解析 ===
START_TIME = parse_time_string(START_TIME_STR)
END_TIME = parse_time_string(END_TIME_STR)
DURATION = END_TIME - START_TIME
if DURATION <= 0:
    raise ValueError("END_TIME 必须晚于 START_TIME")


# === 初始化 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

SAMPLE_RATE = 22050
N_MELS = 64
HOP_LENGTH = 512

# === 加载音频 ===
print("加载音频...")
waveform, sr = torchaudio.load(AUDIO_FILE)
waveform = waveform.mean(dim=0).unsqueeze(0).to(device)  # 转为单声道 + 送入 GPU

# === 计算 Mel 频谱的函数 ===
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=HOP_LENGTH, n_mels=N_MELS
).to(device)

db_transform = T.AmplitudeToDB(top_db=80).to(device)


def extract_mel_db(wav):
    mel = mel_transform(wav)
    mel_db = db_transform(mel)
    return mel_db.squeeze(0)


# === 提取参考片段特征 ===
start_sample = int(START_TIME * SAMPLE_RATE)
end_sample = int((START_TIME + DURATION) * SAMPLE_RATE)
target_segment = waveform[:, start_sample:end_sample]
target_mel = extract_mel_db(target_segment)
target_vector = target_mel.flatten().cpu().numpy()

# === 滑动窗口匹配 ===
print("开始匹配...")
window_size = end_sample - start_sample
step_samples = int(STEP_SIZE * SAMPLE_RATE)
matches = []

for i in tqdm(range(0, waveform.shape[1] - window_size, step_samples)):
    segment = waveform[:, i : i + window_size]
    mel = extract_mel_db(segment)

    if mel.shape != target_mel.shape:
        continue  # 跳过尺寸不一致的情况

    vector = mel.flatten().cpu().numpy()
    similarity = cosine(target_vector, vector)

    if similarity < THRESHOLD:
        match_time = i / SAMPLE_RATE
        matches.append((match_time, similarity))

# === 输出结果 ===
print(f"\n匹配完成（相似度 < {THRESHOLD}）")
filtered_matches = []
last_time = -9999  # 初始化为极小值
for t, s in matches:
    if t - last_time >= INTERVAL:
        filtered_matches.append((t, s))
        last_time = t

for t, s in filtered_matches:
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    print(f"- 匹配时间点：{hours:02d}:{minutes:02d}:{seconds:02d}，相似度：{s:.4f}")

# # === 自动分割音频 ===
# import os
# def ffmpeg_cut(input_file, start, end, output_file):
#     cmd = f'ffmpeg -y -i "{input_file}" -ss {start:.3f} -to {end:.3f} -c copy "{output_file}"'
#     os.system(cmd)

# segments = []
# input_ext = os.path.splitext(AUDIO_FILE)[-1]
# for idx in range(len(filtered_matches)):
#     # 当前片段起止
#     start_cut = filtered_matches[idx][0] + DURATION
#     if idx + 1 < len(filtered_matches):
#         end_cut = filtered_matches[idx+1][0]
#     else:
#         end_cut = waveform.shape[1] / SAMPLE_RATE
#     if end_cut > start_cut:
#         segments.append((start_cut, end_cut))

# for i, (start, end) in enumerate(segments, 1):
#     out_file = f"output_{i:02d}{input_ext}"
#     ffmpeg_cut(AUDIO_FILE, start, end, out_file)
#     print(f"已保存片段: {out_file} [{start:.2f}s ~ {end:.2f}s]")
