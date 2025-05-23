#include <Servo.h>

// Create servo objects
Servo servo1;
Servo servo2;
Servo servo3;
Servo actuator;
Servo actuator2;

// Initial servo PWM positions
int currentPWM1 = 700;
int currentPWM2 = 2200;
int currentPWM3 = 2300;

// Actuator positions (PWM)
int finger_1_open = 700;
int finger_1_close = 2200;
int finger_2_open = 2100;
int finger_2_close = 700;
//int current_position = 1000;
int stepsize = 18 ;
int stepDelay = 40 ;

void setup() {
  // Attach servos
  servo1.attach(6);
  servo2.attach(4);
  servo3.attach(3);
  actuator.attach(9);
  actuator2.attach(8);

  // Set initial positions
  servo1.writeMicroseconds(currentPWM1);
  servo2.writeMicroseconds(currentPWM2);
  servo3.writeMicroseconds(currentPWM3);
  actuator.writeMicroseconds(finger_1_open);
  actuator2.writeMicroseconds(finger_2_open);

  Serial.begin(115200);
  while (!Serial);
  //Serial.println("Arduino is ready");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    //Serial.print("Received command: ");
    //Serial.println(command);

    int commaIndex1 = command.indexOf(',');
    int commaIndex2 = command.indexOf(',', commaIndex1 + 1);

    if (commaIndex1 != -1 && commaIndex2 != -1) {
      // PWM command
      int targetAngle1 = command.substring(0, commaIndex1).toInt();
      int targetAngle2 = command.substring(commaIndex1 + 1, commaIndex2).toInt();
      int targetAngle3 = command.substring(commaIndex2 + 1).toInt();

      //Serial.print("Parsed angles: ");
      //Serial.print(targetAngle1); Serial.print(", ");
      //Serial.print(targetAngle2); Serial.print(", ");
      //Serial.println(targetAngle3);

      int pwmTarget1 = mapAngleToPWM(targetAngle1, 0, 90, 660, 1750);
      int pwmTarget2 = mapAngleToPWM(targetAngle2, -110, 0, 1300, 2150);
      int pwmTarget3 = mapAngleToPWM(targetAngle3, -90, 0, 1150, 2200);

      //movePWMServosSmoothly(pwmTarget1, pwmTarget2, pwmTarget3, stepDelay);
      servo1.writeMicroseconds(pwmTarget1);
      servo2.writeMicroseconds(pwmTarget2);
      servo3.writeMicroseconds(pwmTarget3);

      currentPWM1 = pwmTarget1;
      currentPWM2 = pwmTarget2;
      currentPWM3 = pwmTarget3;

      //Serial.println("Command executed");
    } 
    else {
      // Actuator command
      int actuatorState = command.toInt();

      if (actuatorState == 0) {
        actuator.writeMicroseconds(finger_1_open);
        actuator2.writeMicroseconds(finger_2_open);
        //current_position = finger_1_open;
        //Serial.println("Actuators moved to open");
      } 
      else if (actuatorState == 1) {
        actuator.writeMicroseconds(finger_1_close);
        actuator2.writeMicroseconds(finger_2_close);
        //current_position = finger_1_close;
        //Serial.println("Actuators moved to close");
      } 
      else {
        //Serial.println("Error: Invalid actuator state");
      }

      //Serial.println("Actuator command executed");
    }
  }
}

// Map angle to PWM
int mapAngleToPWM(int angle, int angleMin, int angleMax, int pwmMin, int pwmMax) {
  return map(angle, angleMin, angleMax, pwmMin, pwmMax);
}

// Smooth move PWM
void movePWMServosSmoothly(int targetPWM1, int targetPWM2, int targetPWM3, int stepDelay) {
  int currentPWM1 = servo1.readMicroseconds();
  int currentPWM2 = servo2.readMicroseconds();
  int currentPWM3 = servo3.readMicroseconds();

  int maxSteps = max(abs(targetPWM1 - currentPWM1),
                 max(abs(targetPWM2 - currentPWM2),
                     abs(targetPWM3 - currentPWM3)));

  for (int step = 0; step <= maxSteps; step++) {
    if (currentPWM1 != targetPWM1) {
      currentPWM1 += (targetPWM1 > currentPWM1) ? stepsize : -stepsize;
      servo1.writeMicroseconds(currentPWM1);
    }

    if (currentPWM2 != targetPWM2) {
      currentPWM2 += (targetPWM2 > currentPWM2) ? stepsize : -stepsize;
      servo2.writeMicroseconds(currentPWM2);
    }

    if (currentPWM3 != targetPWM3) {
      currentPWM3 += (targetPWM3 > currentPWM3) ? stepsize : -stepsize;
      servo3.writeMicroseconds(currentPWM3);
    }

    delay(stepDelay);
  }
}
