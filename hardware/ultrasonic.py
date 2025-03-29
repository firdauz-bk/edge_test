import RPi.GPIO as GPIO
import time

class UltrasonicSensor:
    def __init__(self, trigger_pin, echo_pin):
        self.TRIGGER = trigger_pin
        self.ECHO = echo_pin
        self.setup_gpio()
        
    def setup_gpio(self):
        """Setup GPIO pins for the ultrasonic sensor"""
        # Avoid setting mode if already set (to prevent conflicts)
        if GPIO.getmode() is None:
            GPIO.setmode(GPIO.BCM)
            
        GPIO.setup(self.TRIGGER, GPIO.OUT)
        GPIO.setup(self.ECHO, GPIO.IN)
        GPIO.output(self.TRIGGER, False)
        time.sleep(0.5)  # Sensor stabilization

    def get_distance(self):
        """Measure distance in centimeters using the ultrasonic sensor"""
        # Send pulse
        GPIO.output(self.TRIGGER, True)
        time.sleep(0.00001)
        GPIO.output(self.TRIGGER, False)

        # Measure echo duration
        pulse_start = time.time()
        pulse_end = time.time()
        
        timeout = time.time() + 0.04  # 40ms timeout (~7m max range)
        
        # Wait for echo to start
        while GPIO.input(self.ECHO) == 0 and time.time() < timeout:
            pulse_start = time.time()
            
        # Wait for echo to end
        while GPIO.input(self.ECHO) == 1 and time.time() < timeout:
            pulse_end = time.time()

        # Calculate distance
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # Calculate distance in cm
        
        # Limit the range to realistic values
        if distance < 2 or distance > 400:
            return 400  # Out of range or invalid reading
            
        return round(distance, 2)

    def cleanup(self):
        """Clean up GPIO pins used by the sensor"""
        # Clean up only the pins we're using
        try:
            GPIO.cleanup([self.TRIGGER, self.ECHO])
        except:
            # If pins already cleaned up, just pass
            pass

if __name__ == "__main__":
    try:
        sensor = UltrasonicSensor(trigger_pin=17, echo_pin=27)
        print("Ultrasonic Sensor Test. Press Ctrl+C to exit.")
        
        while True:
            dist = sensor.get_distance()
            print(f"Distance: {dist} cm")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Test stopped by user")
        sensor.cleanup()