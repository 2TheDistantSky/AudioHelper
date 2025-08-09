import torch
import torchaudio
import torchaudio.transforms as T
from scipy.spatial.distance import cosine
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)


# === 配置区域 ===
AUDIO_FILE = "dataset/your_audio.mp3"
START_TIME_STR = "0:0:6"
END_TIME_STR = "0:1:40"
STEP_SIZE = 0.1  # 滑动窗口步长，单位：秒
THRESHOLD = 0.06  # 相似度阈值（越小越严格）
INTERVAL = 10  # 秒
N_MELS = 64
HOP_LENGTH = 512


class AudioMatcher:
    def __init__(
        self,
        audio_file,
        start_time_str,
        end_time_str,
        step_size=0.1,
        threshold=0.06,
        interval=10,
        n_mels=64,
        hop_length=512,
    ):
        self.audio_file = audio_file
        self.start_time = self.parse_time_string(start_time_str)
        self.end_time = self.parse_time_string(end_time_str)
        self.duration = self.end_time - self.start_time
        if self.duration <= 0:
            raise ValueError("END_TIME 必须晚于 START_TIME")
        self.step_size = step_size
        self.threshold = threshold
        self.interval = interval
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        self._init_audio_info()
        self._init_transforms()
        self._load_waveform()
        self.target_vector = self._extract_target_vector()

    @staticmethod
    def parse_time_string(time_str: str) -> int:
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

    def _init_audio_info(self):
        print("获取音频信息...")
        info = torchaudio.info(self.audio_file)
        self.num_frames = info.num_frames
        self.sample_rate = info.sample_rate
        print(f"音频采样率: {self.sample_rate}")

    def _init_transforms(self):
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        ).to(self.device)
        self.db_transform = T.AmplitudeToDB(top_db=80).to(self.device)

    def _load_waveform(self):
        print("加载音频到内存...")
        waveform, sr = torchaudio.load(self.audio_file)
        waveform = waveform.mean(dim=0).unsqueeze(0).to(self.device)
        self.waveform = waveform
        self.total_samples = waveform.shape[1]

    def extract_mel_db(self, wav):
        mel = self.mel_transform(wav)
        mel_db = self.db_transform(mel)
        return mel_db.squeeze(0)

    def _extract_target_vector(self):
        start_sample = int(self.start_time * self.sample_rate)
        end_sample = int((self.start_time + self.duration) * self.sample_rate)
        target_segment = self.waveform[:, start_sample:end_sample]
        target_mel = self.extract_mel_db(target_segment)
        return target_mel.flatten().cpu().numpy()

    def match(self):
        print("开始匹配...")
        window_size = int(self.duration * self.sample_rate)
        step_samples = int(self.step_size * self.sample_rate)
        matches = []
        for i in tqdm(range(0, self.total_samples - window_size, step_samples)):
            segment = self.waveform[:, i : i + window_size]
            mel = self.extract_mel_db(segment)
            if mel.shape != (self.n_mels, mel.shape[1]):
                continue
            vector = mel.flatten().cpu().numpy()
            similarity = cosine(self.target_vector, vector)
            if similarity < self.threshold:
                match_time = i / self.sample_rate
                matches.append((match_time, similarity))
        return matches

    def print_results(self, matches):
        print(f"\n匹配完成（相似度 < {self.threshold}）")
        filtered_matches = []
        last_time = -9999
        for t, s in matches:
            if t - last_time >= self.interval:
                filtered_matches.append((t, s))
                last_time = t
        for t, s in filtered_matches:
            hours = int(t // 3600)
            minutes = int((t % 3600) // 60)
            seconds = int(t % 60)
            print(
                f"- 匹配时间点：{hours:02d}:{minutes:02d}:{seconds:02d}，相似度：{s:.4f}"
            )


class AudioCutter:
    def __init__(self, audio_file, sample_rate):
        self.audio_file = audio_file
        self.sample_rate = sample_rate
        self.output_dir = "output_segments"
        os.makedirs(self.output_dir, exist_ok=True)

    def cut(self, matches, duration, total_samples):
        # 计算分割区间，去掉匹配段，保留不匹配段
        segments = []
        match_times = [t for t, _ in matches]
        match_times.append(total_samples / self.sample_rate)  # 结尾
        prev_end = 0.0
        for i, t in enumerate(match_times[:-1]):
            start_cut = t + duration
            end_cut = match_times[i + 1]
            if end_cut > start_cut:
                segments.append((start_cut, end_cut))
        return segments

    def save_segments(self, waveform, segments):
        for idx, (start, end) in enumerate(segments, 1):
            start_sample = int(start * self.sample_rate)
            end_sample = int(end * self.sample_rate)
            segment_waveform = waveform[:, start_sample:end_sample]
            out_path = os.path.join(self.output_dir, f"segment_{idx:02d}.wav")
            torchaudio.save(out_path, segment_waveform.cpu(), self.sample_rate)
            print(f"已保存片段: {out_path} [{start:.2f}s ~ {end:.2f}s]")


# === 主流程 ===
if __name__ == "__main__":
    matcher = AudioMatcher(
        audio_file=AUDIO_FILE,
        start_time_str=START_TIME_STR,
        end_time_str=END_TIME_STR,
        step_size=STEP_SIZE,
        threshold=THRESHOLD,
        interval=INTERVAL,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
    )
    matches = matcher.match()
    matcher.print_results(matches)

    # 分割音频
    cutter = AudioCutter(AUDIO_FILE, matcher.sample_rate)
    segments = cutter.cut(matches, matcher.duration, matcher.total_samples)
    cutter.save_segments(matcher.waveform, segments)
