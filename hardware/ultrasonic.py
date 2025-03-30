import RPi.GPIO as GPIO
import time

class UltrasonicSensor:
    def __init__(self, trigger_pin, echo_pin):
        """Initialize the ultrasonic sensor with specified pins.
        
        Args:
            trigger_pin: GPIO pin for the trigger signal
            echo_pin: GPIO pin for the echo signal
        """
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        
        # Set up GPIO pins
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trigger_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
        
        # Initialize trigger as low
        GPIO.output(self.trigger_pin, False)
        time.sleep(0.5)  # Let the sensor settle
        
    def get_distance(self):
        """Get the distance measurement from the sensor in centimeters.
        
        Returns:
            float: Distance in centimeters, or float('inf') if measurement failed
        """
        # Send 10us pulse to trigger
        GPIO.output(self.trigger_pin, True)
        time.sleep(0.00001)  # 10µs pulse
        GPIO.output(self.trigger_pin, False)
        
        # Wait for echo to start
        pulse_start = time.time()
        timeout_start = time.time()
        
        # Set timeout for 1 second
        while GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()
            if time.time() - timeout_start > 1:
                return float('inf')  # Return infinity if timed out
        
        # Wait for echo to end
        timeout_start = time.time()
        pulse_end = time.time()
        
        while GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()
            if time.time() - timeout_start > 1:
                return float('inf')  # Return infinity if timed out
        
        # Calculate distance
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # Speed of sound is 343m/s, distance = time * speed / 2
        distance = round(distance, 2)
        
        return distance