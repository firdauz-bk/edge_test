import RPi.GPIO as GPIO
import time

class DoorLock:
    def __init__(self, relay_pin=14):
        self.RELAY_PIN = relay_pin
        self.setup()
        
    def setup(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.RELAY_PIN, GPIO.OUT)
        GPIO.output(self.RELAY_PIN, GPIO.LOW)  # Start in locked state
        print("Door Lock System Initialized")
        
    def lock(self):
        print("Locking door...")
        GPIO.output(self.RELAY_PIN, GPIO.LOW)
        print("Door Locked")
        
    def unlock(self):
        print("Unlocking door...")
        GPIO.output(self.RELAY_PIN, GPIO.HIGH)
        print("Door Unlocked")
        
    def test_cycle(self):
        print("Running a test cycle...")
        self.lock()
        time.sleep(2)
        
        self.unlock()
        time.sleep(5)
        
        self.lock()
        print("Test cycle complete")
        
    def cleanup(self):
        self.lock()  # Ensure door is locked when shutting down
        GPIO.cleanup(self.RELAY_PIN)
        print("Door lock GPIO cleanup complete")

if __name__ == "__main__":
    lock_system = DoorLock()
    try:
        while True:
            command = input("\nEnter Command (unlock/lock/test/exit): ").lower()
            if command == "unlock":
                lock_system.unlock()
            elif command == "lock":
                lock_system.lock()
            elif command == "test":
                lock_system.test_cycle()
            elif command == "exit":
                break
            else:
                print("Invalid command")
                
    except KeyboardInterrupt:
        print("\nProgram stopped")
    finally:
        lock_system.cleanup()