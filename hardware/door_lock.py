import RPi.GPIO as GPIO
import time

class DoorLock:
    def __init__(self, solenoid_pin=14):
        """
        Initialize the door lock controller
        
        Args:
            solenoid_pin (int): GPIO pin connected to the door lock solenoid
        """
        self.SOLENOID_PIN = solenoid_pin
        self.setup_gpio()
        
    def setup_gpio(self):
        """Setup GPIO pins for the door lock solenoid"""
        # Set GPIO mode if not already set
        if GPIO.getmode() is None:
            GPIO.setmode(GPIO.BCM)
            
        # Set up solenoid pin
        GPIO.setup(self.SOLENOID_PIN, GPIO.OUT)
        GPIO.output(self.SOLENOID_PIN, GPIO.LOW)  # Ensure door is locked initially
        
    def unlock(self):
        """Unlock the door by activating the solenoid"""
        GPIO.output(self.SOLENOID_PIN, GPIO.HIGH)
        print("Door unlocked")
        
    def lock(self):
        """Lock the door by deactivating the solenoid"""
        GPIO.output(self.SOLENOID_PIN, GPIO.LOW)
        print("Door locked")
        
    def cleanup(self):
        """Clean up GPIO pins"""
        try:
            GPIO.cleanup(self.SOLENOID_PIN)
        except:
            # If already cleaned up, just pass
            pass