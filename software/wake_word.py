import torch
import torch.nn as nn
import torchaudio.transforms as T
import sounddevice as sd
import numpy as np
import time

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

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# --- Audio Processing Functions ---
def process_audio(audio_data, device):
    transform = T.MelSpectrogram(
        sample_rate=16000, 
        n_mels=64, 
        n_fft=400, 
        hop_length=160
    ).to(device)
    
    waveform = torch.tensor(audio_data, dtype=torch.float32).to(device)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # Ensure batch dimension
    mel_spec = transform(waveform).unsqueeze(0)  # Add channel dimension for Conv2D
    return mel_spec

def detect_wake_word(model, audio_data, device):
    mel_spec = process_audio(audio_data, device)
    with torch.no_grad():
        output = model(mel_spec)
        return torch.sigmoid(output).item()  # Sigmoid for probability
    
        # When detection happens, put an event in the queue
    if prediction > THRESHOLD:
        # Stop the ultrasonic sensor thread
        stop_audio_stream()
        
        # Notify the main application
        event_queue.put(("WAKEUP_WORD_DETECTED", None))
        
        print("Wake word detected!")
    
    return prediction

# --- Audio Stream Control ---
def set_callback(callback):
    global callback_function
    callback_function = callback

def start_audio_stream():
    global audio_stream
    SAMPLE_RATE = 16000
    DURATION = 1.0
    BUFFER_SIZE = int(SAMPLE_RATE * DURATION)
    
    # Wait a bit to ensure resources are released
    time.sleep(0.5)
    
    try:
        audio_stream = sd.InputStream(
            callback=callback_function, 
            channels=1, 
            samplerate=SAMPLE_RATE, 
            blocksize=BUFFER_SIZE,
            device=0,  # Explicitly specify device
            latency='high',  # Use higher latency for stability
            dtype='float32'
        )
        audio_stream.start()
        print("Listening for wake word...")
    except Exception as e:
        print(f"Error starting audio stream: {e}")
        # Try to reset sounddevice
        try:
            sd._terminate()
            time.sleep(1)
            sd._initialize()
            time.sleep(0.5)
            print("Attempted to reset sounddevice")
            
            # Try again with different device
            audio_stream = sd.InputStream(
                callback=callback_function, 
                channels=1, 
                samplerate=SAMPLE_RATE, 
                blocksize=BUFFER_SIZE,
                latency='high',
                dtype='float32'
            )
            audio_stream.start()
            print("Second attempt successful")
        except Exception as recovery_error:
            print(f"Recovery failed: {recovery_error}")

def stop_audio_stream():
    global audio_stream
    if audio_stream is not None:
        try:
            audio_stream.stop()
            audio_stream.close()
            print("Audio stream stopped.")
        except Exception as e:
            print(f"Error closing audio stream: {e}")
        finally:
            audio_stream = None  # Clear the stream object
            
            # Force sounddevice to release resources
            try:
                sd._terminate()
                time.sleep(0.5)
                sd._initialize()
                time.sleep(0.5)
                print("Sounddevice resources reset")
            except Exception as e:
                print(f"Error resetting sounddevice: {e}")

# In wake_word.py, add these functions:
def pause_audio_stream():
    global audio_stream
    if audio_stream is not None and audio_stream.active:
        try:
            audio_stream.stop()
            print("Audio stream paused.")
        except Exception as e:
            print(f"Error pausing audio stream: {e}")
            stop_audio_stream()  # Fall back to full stop if pause fails

def resume_audio_stream():
    global audio_stream
    if audio_stream is not None and not audio_stream.active:
        try:
            audio_stream.start()
            print("Audio stream resumed.")
        except Exception as e:
            print(f"Error resuming audio stream: {e}")
            # Try to recreate the stream
            stop_audio_stream()
            time.sleep(1)
            start_audio_stream()