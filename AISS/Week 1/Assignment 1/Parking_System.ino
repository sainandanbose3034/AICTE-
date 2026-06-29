#include <ESP32Servo.h>
Servo myServo;
const int red=27;
const int green=26;
const int blue=25;
const int pot=32;
const int servo=33;
void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial.println("Hello, ESP32!");
  pinMode(red,OUTPUT);
  pinMode(green,OUTPUT);
  pinMode(blue,OUTPUT);
  myServo.attach(servo);
}
void setColor(int r,int g,int b){
    analogWrite(red,r);
    analogWrite(blue,b);
    analogWrite(green,g);
}
void loop() {
  int pot_i=analogRead(pot);
  if(pot_i>=0 && pot_i<=1365){
    setColor(255,0,0);
    myServo.write(0);
    Serial.println("Parking Full");
    Serial.println("Limited Parking");
  }
  else if(pot_i>=1366 && pot_i<=2730){
    setColor(255,255,0);
    myServo.write(90);
  }
  else{
    setColor(0,255,0);
    myServo.write(180);
  }
}
