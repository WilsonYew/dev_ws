/***************************************************************
   Motor driver function definitions - by James Nugen
   *************************************************************/

#ifdef L298_MOTOR_DRIVER
  #define RIGHT_MOTOR_DIRECTION 10
  #define RIGHT_MOTOR_SPEED 11
  #define LEFT_MOTOR_DIRECTION 8
  #define LEFT_MOTOR_SPEED 9
  // #define RIGHT_MOTOR_ENABLE 12
  // #define LEFT_MOTOR_ENABLE 13
  #define MIN_PWM 130 //the wheel too heavy need atleast 130 pwm to move
#endif

void initMotorController();
void setMotorSpeed(int i, int spd);
void setMotorSpeeds(int leftSpeed, int rightSpeed);
