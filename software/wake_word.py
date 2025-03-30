import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
import sounddevice as sd
import threading
import time

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

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class WakeWordDetector:
    def __init__(self, model_path="476998.pth", threshold=0.78):
        # Audio configuration
        self.SAMPLE_RATE = 16000
        self.DURATION = 1.0
        self.BUFFER_SIZE = int(self.SAMPLE_RATE * self.DURATION)
        self.THRESHOLD = threshold
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = WakeWordModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Set up mel spectrogram transform
        self.transform = T.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE, 
            n_mels=64, 
            n_fft=400, 
            hop_length=160
        ).to(self.device)
        
        # Initialize audio buffer
        self.audio_buffer = np.zeros(self.BUFFER_SIZE, dtype=np.float32)
        
        # Audio stream
        self.audio_stream = None
        self.running = False
        self.callback_fn = None
    
    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio callback error: {status}")
            
        # Update the buffer
        self.audio_buffer[:-frames] = self.audio_buffer[frames:]
        self.audio_buffer[-frames:] = indata[:, 0]
        
        # Detect wake word
        prediction = self.detect_wake_word(self.audio_buffer)
        print(f"Wake word probability: {prediction:.4f}")
        
        # If wake word detected, trigger callback
        if prediction > self.THRESHOLD and self.callback_fn:
            self.stop()  # Stop listening
            self.callback_fn()  # Trigger callback
    
    def process_audio(self, audio_data):
        """Process audio data to create mel spectrogram"""
        waveform = torch.tensor(audio_data, dtype=torch.float32).to(self.device)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # Ensure batch dimension
        mel_spec = self.transform(waveform).unsqueeze(0)  # Add channel dimension for Conv2D
        return mel_spec
    
    def detect_wake_word(self, audio_data):
        """Detect wake word in audio data"""
        mel_spec = self.process_audio(audio_data)
        with torch.no_grad():
            output = self.model(mel_spec)
            return torch.sigmoid(output).item()  # Sigmoid for probability
    
    def start(self, callback=None):
        """Start wake word detection"""
        if self.running:
            return
            
        self.callback_fn = callback
        self.running = True
        
        try:
            # Set default sound device if not specified
            sd.default.device = 0
            
            # Start the audio stream
            self.audio_stream = sd.InputStream(
                callback=self.audio_callback, 
                channels=1, 
                samplerate=self.SAMPLE_RATE, 
                blocksize=self.BUFFER_SIZE
            )
            self.audio_stream.start()
            print("Wake word detection started")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.running = False
    
    def stop(self):
        """Full cleanup of audio resources"""
        if self.audio_stream is not None:
            if self.audio_stream.active:
                self.audio_stream.stop()
            self.audio_stream.close()
        sd._terminate()  # Force terminate PortAudio
        self.running = False
        self.audio_stream = None
        

# Test function
def main():
    detector = WakeWordDetector()
    
    def on_wake_word():
        print("Wake word detected!")
        detector.stop()
    
    detector.start(callback=on_wake_word)
    
    try:
        # Run for 30 seconds
        for _ in range(30):
            time.sleep(1)
            if not detector.running:
                break
    finally:
        detector.stop()

if __name__ == "__main__":
    main()