import torch
import torch.nn as nn
import torchaudio.transforms as T
import sounddevice as sd
import numpy as np

# Global variables
audio_stream = None
callback_function = None

# --- Wake Word Model ---
class WakeWordModel(nn.Module):
    def __init__(self):
        super(WakeWordModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)  # Dropout to reduce overfitting
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Audio configuration
        self.SAMPLE_RATE = 16000
        self.transform = T.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE, 
            n_mels=64, 
            n_fft=400, 
            hop_length=160
        ).to(self.device)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    def process_audio(self, audio_data):
        waveform = torch.tensor(audio_data, dtype=torch.float32).to(self.device)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # Ensure batch dimension
        mel_spec = self.transform(waveform).unsqueeze(0)  # Add channel dimension for Conv2D
        return mel_spec
    
    def detect_wake_word(self, audio_data):
        mel_spec = self.process_audio(audio_data)
        with torch.no_grad():
            output = self(mel_spec)
            return torch.sigmoid(output).item()  # Sigmoid for probability

# --- Audio Stream Control ---
def set_callback(callback):
    global callback_function
    callback_function = callback

def start_audio_stream():
    global audio_stream
    SAMPLE_RATE = 16000
    DURATION = 1.0
    BUFFER_SIZE = int(SAMPLE_RATE * DURATION)
    
    try:
        audio_stream = sd.InputStream(
            callback=callback_function, 
            channels=1, 
            samplerate=SAMPLE_RATE, 
            blocksize=BUFFER_SIZE
        )
        audio_stream.start()
        print("Listening for wake word...")
    except Exception as e:
        print(f"Error starting audio stream: {e}")

def stop_audio_stream():
    global audio_stream
    if audio_stream is not None:
        audio_stream.stop()
        audio_stream.close()
        audio_stream = None  # Clear the stream object
        print("Audio stream stopped.")