import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
import sounddevice as sd

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
        self.dropout = nn.Dropout(0.5)
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
    def __init__(self, callback, threshold=0.78, model_path="476998.pth", sample_rate=16000, duration=1.0):
        self.callback = callback  # Function to call when wake word is detected
        self.SAMPLE_RATE = sample_rate
        self.DURATION = duration
        self.BUFFER_SIZE = int(self.SAMPLE_RATE * self.DURATION)
        self.THRESHOLD = threshold
        
        # Initialize audio buffer
        self.audio_buffer = np.zeros(self.BUFFER_SIZE, dtype=np.float32)
        
        # Initialize audio stream
        self.audio_stream = None
        self.is_running = False
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = WakeWordModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Transform for mel spectrogram
        self.transform = T.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE, 
            n_mels=64, 
            n_fft=400, 
            hop_length=160
        ).to(self.device)
    
    def audio_callback(self, indata, frames, time, status):
        """Callback function for the audio stream"""
        if status:
            print(f"Audio callback error: {status}")

        # Update audio buffer
        self.audio_buffer[:-frames] = self.audio_buffer[frames:]
        self.audio_buffer[-frames:] = indata[:, 0]
        
        # Detect wake word
        prediction = self.detect_wake_word(self.audio_buffer)
        print(f"Wake word probability: {prediction}")

        if prediction > self.THRESHOLD:
            print("Wake word detected!")
            self.stop()  # Stop listening
            self.callback()  # Call the provided callback function
    
    def process_audio(self, audio_data):
        """Process audio data into mel spectrogram for model input"""
        waveform = torch.tensor(audio_data, dtype=torch.float32).to(self.device)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # Ensure batch dimension
        mel_spec = self.transform(waveform).unsqueeze(0)  # Add channel dimension for Conv2D
        return mel_spec
    
    def detect_wake_word(self, audio_data):
        """Process audio and get wake word detection probability"""
        mel_spec = self.process_audio(audio_data)
        with torch.no_grad():
            output = self.model(mel_spec)
            return torch.sigmoid(output).item()  # Sigmoid for probability
    
    def start(self):
        """Start listening for wake word"""
        if not self.is_running:
            try:
                # Set sounddevice default device
                sd.default.device = 0
                
                # Start audio stream
                self.audio_stream = sd.InputStream(
                    callback=self.audio_callback,
                    channels=1,
                    samplerate=self.SAMPLE_RATE,
                    blocksize=self.BUFFER_SIZE
                )
                self.audio_stream.start()
                self.is_running = True
                print("Listening for wake word...")
            except Exception as e:
                print(f"Error starting audio stream: {e}")
    
    def stop(self):
        """Stop listening for wake word"""
        if self.is_running and self.audio_stream is not None:
            try:
                self.is_running = False  # Set this first
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
                print("Wake word detection stopped.")
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
                # Force reset of the stream object
                self.audio_stream = None
                self.is_running = False