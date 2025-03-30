import RPi.GPIO as GPIO

class DoorLock:
    def __init__(self, relay_pin=14):
        self.relay_pin = relay_pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.relay_pin, GPIO.OUT)
        GPIO.output(self.relay_pin, GPIO.HIGH)  # Assuming relay is active LOW
        print("Door lock system initialized")
    
    def unlock(self):
        print("Unlocking door")
        GPIO.output(self.relay_pin, GPIO.LOW)  # Activate relay
    
    def lock(self):
        print("Locking door")
        GPIO.output(self.relay_pin, GPIO.HIGH)  # Deactivate relay
    
    def cleanup(self):
        GPIO.cleanup()
        print("GPIO cleanup complete")