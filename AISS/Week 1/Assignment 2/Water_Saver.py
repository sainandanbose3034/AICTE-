from machine import Pin, ADC, PWM
import time
#Smart Water Monitor
pot = ADC(Pin(34))
pot.atten(ADC.ATTN_11DB)

led_r = Pin(25, Pin.OUT)
led_g = Pin(26, Pin.OUT)
led_b = Pin(27, Pin.OUT)

servo = PWM(Pin(15), freq=50)

def set_color(r, g, b):
    led_r.value(r)
    led_g.value(g)
    led_b.value(b)

def set_servo_angle(angle):
    if angle == 0:
        servo.duty(26)
    elif angle == 90:
        servo.duty(78)
    elif angle == 180:
        servo.duty(127)

while True:
    pot_value = pot.read()
    water_level = (pot_value / 4095) * 100
    
    print(f"Water Level: {water_level:.1f}%")

    if water_level <= 33:
        set_color(1, 0, 0)
        set_servo_angle(180)
    elif water_level <= 66:
        set_color(1, 1, 0)
        set_servo_angle(90)
    else:
        set_color(0, 1, 0)
        set_servo_angle(0)
        
    time.sleep(0.2)
