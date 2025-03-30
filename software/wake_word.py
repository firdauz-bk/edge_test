import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
import sounddevice as sd
import threading
import queue
import time

class WakeWordModel(nn.Module):
    """Neural network model for wake word detection"""
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
    def __init__(self, model_path="476998.pth", threshold=0.78, event_queue=None):
        """
        Initialize the wake word detector
        
        Args:
            model_path (str): Path to the trained model weights
            threshold (float): Confidence threshold for detection
            event_queue (queue.Queue): Queue for sending events to main thread
        """
        self.SAMPLE_RATE = 16000
        self.DURATION = 1.0
        self.BUFFER_SIZE = int(self.SAMPLE_RATE * self.DURATION)
        self.THRESHOLD = threshold
        self.event_queue = event_queue
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = WakeWordModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Audio transform
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
        self.is_listening = False
        
    def process_audio(self, audio_data):
        """Process audio data to prepare for model input"""
        waveform = torch.tensor(audio_data, dtype=torch.float32).to(self.device)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # Ensure batch dimension
        mel_spec = self.transform(waveform).unsqueeze(0)  # Add channel dimension for Conv2D
        return mel_spec
        
    def detect_wake_word(self, audio_data):
        """Run audio through the model to detect wake word"""
        mel_spec = self.process_audio(audio_data)
        with torch.no_grad():
            output = self.model(mel_spec)
            return torch.sigmoid(output).item()  # Sigmoid for probability
            
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream - processes incoming audio"""
        if status:
            print(f"Audio callback error: {status}")
            return
            
        # Update buffer with new audio data
        self.audio_buffer[:-frames] = self.audio_buffer[frames:]
        self.audio_buffer[-frames:] = indata[:, 0]
        
        # Run detection on buffer
        prediction = self.detect_wake_word(self.audio_buffer)
        
        # Debug print every second
        if frames == self.BUFFER_SIZE:
            print(f"Wake word probability: {prediction:.4f}")
            
        # If prediction exceeds threshold, trigger wake word event
        if prediction > self.THRESHOLD:
            print("Wake word detected!")
            self.stop_listening()
            
            # Send event to main thread
            if self.event_queue:
                self.event_queue.put(("WAKEUP_WORD_DETECTED", None))
                
    def start_listening(self):
        """Start the audio stream for wake word detection"""
        if self.is_listening:
            return
            
        try:
            self.audio_stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.SAMPLE_RATE,
                blocksize=int(self.SAMPLE_RATE * 0.1)  # Process in smaller chunks (100ms)
            )
            self.audio_stream.start()
            self.is_listening = True
            print("Wake word detector listening...")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            
    def stop_listening(self):
        """Stop the audio stream"""
        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
            self.is_listening = False
            print("Wake word detector stopped.")
            
    def cleanup(self):
        """Clean up resources"""
        self.stop_listening()