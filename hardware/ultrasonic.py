import RPi.GPIO as GPIO
import time
import math

class KalmanFilter:
    def __init__(self, q=1.0, r=0.5, p=1.0, initial_value=20.0):
        """
        Initialize Kalman filter with tuned parameters for ultrasonic sensors.
        
        Args:
            q (float): Process noise (lower for more stable output)
            r (float): Measurement noise (higher for noisier measurements)
            p (float): Initial estimation error covariance
            initial_value (float): Initial state estimate
        """
        self.q = q  # Process noise
        self.r = r  # Measurement noise
        self.p = p  # Estimation error covariance
        self.x = initial_value  # State estimate
        self.k = 0  # Kalman gain
        
    def update(self, measurement):
        """
        Update Kalman filter with a new measurement.
        
        Args:
            measurement (float): New measurement value
            
        Returns:
            float: Updated state estimate
        """
        # Calculate innovation (measurement - prediction)
        innovation = measurement - self.x
        
        # Adaptive process noise based on innovation
        if abs(innovation) > 10.0:
            # Temporarily increase process noise for large changes
            self.q *= 2.0
        else:
            # Return to normal process noise
            self.q = 1.0
            
        # Prediction update
        self.p = self.p + self.q
        
        # Measurement update
        self.k = self.p / (self.p + self.r)
        
        # More aggressive update for small innovations
        if abs(innovation) < 50.0:
            self.x += self.k * innovation
        else:
            # For very large changes, trust the measurement more
            self.x = 0.7 * measurement + 0.3 * self.x
            
        # Ensure estimate stays within bounds
        if self.x < 2.0:
            self.x = 2.0
        if self.x > 400.0:
            self.x = 400.0
            
        # Update error covariance
        self.p = (1 - self.k) * self.p
        
        return self.x


class UltrasonicSensor:
    # Constants for measurement limits
    MIN_DISTANCE_CM = 2.0
    MAX_DISTANCE_CM = 400.0
    SPEED_OF_SOUND_CM_US = 0.0343  # Speed of sound in cm/microsecond
    MEASUREMENT_TIMEOUT = 0.015  # 15ms timeout (equivalent to ~2.5m)
    
    def __init__(self, trigger_pin, echo_pin, kalman_q=1.0, kalman_r=0.5):
        self.TRIGGER = trigger_pin
        self.ECHO = echo_pin
        self.obstacle_detected = False
        self.last_valid_distance = 20.0  # Initial default distance
        
        # Initialize Kalman filter
        self.kalman = KalmanFilter(q=kalman_q, r=kalman_r, initial_value=self.last_valid_distance)
        
        self.setup_gpio()
        
    def setup_gpio(self):
        """Setup GPIO pins for the ultrasonic sensor"""
        # Avoid setting mode if already set (to prevent conflicts)
        if GPIO.getmode() is None:
            GPIO.setmode(GPIO.BCM)
            
        GPIO.setup(self.TRIGGER, GPIO.OUT)
        GPIO.setup(self.ECHO, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Add pull-down resistor
        GPIO.output(self.TRIGGER, False)
        time.sleep(0.05)  # Shorter stabilization time
    
    def get_pulse_duration(self):
        """
        Measure the echo pulse duration with improved timeout handling.
        
        Returns:
            float: Pulse duration in seconds, or None if measurement invalid
        """
        # Send 10µs pulse
        GPIO.output(self.TRIGGER, True)
        time.sleep(0.00001)  # 10 microseconds
        GPIO.output(self.TRIGGER, False)
        
        # Initialize variables
        pulse_start = None
        pulse_end = None
        timeout_start = time.time()
        
        # Wait for echo to start with timeout
        while GPIO.input(self.ECHO) == 0:
            pulse_start = time.time()
            if time.time() - timeout_start > self.MEASUREMENT_TIMEOUT:
                return None
            
        # Wait for echo to end with timeout
        while GPIO.input(self.ECHO) == 1:
            pulse_end = time.time()
            if time.time() - timeout_start > self.MEASUREMENT_TIMEOUT:
                return None
        
        # Calculate pulse duration
        if pulse_start and pulse_end:
            return pulse_end - pulse_start
        
        return None

    def get_distance(self):
        """
        Measure distance in centimeters with Kalman filtering.
        
        Returns:
            float: Filtered distance in centimeters
        """
        pulse_duration = self.get_pulse_duration()
        
        # Handle invalid measurements
        if pulse_duration is None:
            # Keep previous estimate if measurement is invalid
            self.obstacle_detected = (self.kalman.x < 10.0)
            return self.kalman.x
        
        # Convert to distance
        distance = pulse_duration * 17150  # Calculate distance in cm
        
        # Check if measurement is within valid range
        if distance < self.MIN_DISTANCE_CM or distance > self.MAX_DISTANCE_CM:
            # Keep previous estimate for invalid measurements
            return self.kalman.x
            
        # Apply Kalman filter and update obstacle detection
        filtered_distance = self.kalman.update(distance)
        self.obstacle_detected = (distance < 10.0) or (filtered_distance < 10.0)
        
        return round(filtered_distance, 2)
        
    def is_obstacle_detected(self):
        """
        Returns whether an obstacle is detected (distance < 10cm)
        
        Returns:
            bool: True if obstacle detected, False otherwise
        """
        return self.obstacle_detected

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
        # Create sensor with tuned Kalman parameters
        # Adjust these values based on your sensor's performance:
        # - Increase r (to 0.75 or 1.0) if readings are too jumpy
        # - Increase q (to 1.5 or 2.0) if response is too slow
        # - Decrease q (to 0.75) if readings are too noisy
        sensor = UltrasonicSensor(trigger_pin=17, echo_pin=27, kalman_q=1.0, kalman_r=0.5)
        
        print("Improved Ultrasonic Sensor Test. Press Ctrl+C to exit.")
        
        while True:
            dist = sensor.get_distance()
            obstacle_status = "[OBSTACLE]" if sensor.is_obstacle_detected() else ""
            print(f"Distance: {dist} cm {obstacle_status}")
            time.sleep(0.05)  # 10Hz measurement rate (matches C code)
            
    except KeyboardInterrupt:
        print("Test stopped by user")
        sensor.cleanup()