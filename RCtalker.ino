/*  ___   ___  ___  _   _  ___   ___   ____ ___  ____  
 * / _ \ /___)/ _ \| | | |/ _ \ / _ \ / ___) _ \|    \ 
 *| |_| |___ | |_| | |_| | |_| | |_| ( (__| |_| | | | |
 * \___/(___/ \___/ \__  |\___/ \___(_)____)___/|_|_|_|
 *                  (____/ 
 * Osoyoo Wifi Arduino 2WD Robot Car project
 * USe WI-FI UDP protocol to control robot car
 * tutorial url: https://osoyoo.com/?p=57088
 */
/*Declare L298N Dual H-Bridge Motor Controller directly since there is not a library to load.*/
#include <WiFiEspUdp.h>
WiFiEspUDP Udp;
unsigned int localPort = 8888;  // local port to listen on

//Define L298N Dual H-Bridge Motor Controller Pins
#define RightDirectPin1 12  //Right Motor direction pin 1 to MODEL-X IN1 grey
#define RightDirectPin2 11  //Right Motor direction pin 1 to MODEL-X IN2 yellow
#define speedPinL 6         //Left PWM pin connect MODEL-X ENB brown
#define LeftDirectPin1 7    //Left Motor direction pin 1 to MODEL-X IN3 green
#define LeftDirectPin2 8    //Left Motor direction pin 1 to MODEL-X IN4 white
#define speedPinR 9         // RIGHT PWM pin connect MODEL-X ENA blue
#define SOFT_RX 4           // Softserial RX port
#define SOFT_TX 5           //Softserial TX port

#define FAST_SPEED 180
#define MID_SPEED 130

#define SPEED 120  //avoidance motor speed
#define SPEED_LEFT 125
#define SPEED_RIGHT 125
#define BACK_SPEED 120  //back speed

const int turntime = 300;  //Time the robot spends turning (miliseconds)
const int backtime = 300;  //Time the robot spends turning (miliseconds)


#define MAX_PACKETSIZE 32  //Serial receive buffer
char buffUART[MAX_PACKETSIZE];
unsigned int buffUARTIndex = 0;
unsigned long preUARTTick = 0;

enum DS {
  MANUAL_DRIVE,
  AUTO_DRIVE_LF,  //line follow
  AUTO_DRIVE_UO   //ultrasonic obstruction
} Drive_Status = MANUAL_DRIVE;

enum DN {
  GO_ADVANCE,
  GO_LEFT,
  GO_RIGHT,
  GO_BACK,
  STOP_STOP,
  DEF
} Drive_Num = DEF;
String WorkMode = "?";

//String toggleStr="<form action=\"/\" method=GET><input type=submit name=a value=L><input type=submit name=a value=U><input type=submit name=a value=D><input type=submit name=a value=R></form>";
#define DEBUG true

#include "WiFiEsp.h"
// Emulate Serial1 on pins 9/10 by default
// If you want to use Hard Serial1 in Mega2560 , please remove the wifi shield jumper cap on ESP8266 RX/TX PIN , CONNECT TX->D18 RX->D19
#ifndef HAVE_HWSERIAL1
#include "SoftwareSerial.h"
SoftwareSerial Serial1(SOFT_RX, SOFT_TX);  // RX, TX
#endif

char ssid[] = "SMPREP GUEST";  //"SMPREP STUDENTS 2026"; //"WREN"; //"SMPREP";   // replace *** with your router wifi SSID (name)
char pass[] = "6n35x6n35x";    //"2026Mariner"; //"WREN5678"; //"5W973g@Rax0Vt";   // replace *** with your router wifi password
char packetBuffer[5];
int status = WL_IDLE_STATUS;  // the Wifi radio's status
int connectionId;



// use a ring buffer to increase speed and reduce memory allocation
RingBuffer buf(8);

void go_Advance(void)  //Forward
{
  digitalWrite(RightDirectPin1, HIGH);
  digitalWrite(RightDirectPin2, LOW);
  digitalWrite(LeftDirectPin1, HIGH);
  digitalWrite(LeftDirectPin2, LOW);
  set_Motorspeed(SPEED, SPEED);
}
void go_Left()  //Turn left
{
  digitalWrite(RightDirectPin1, HIGH);
  digitalWrite(RightDirectPin2, LOW);
  digitalWrite(LeftDirectPin1, LOW);
  digitalWrite(LeftDirectPin2, HIGH);
  set_Motorspeed(0, SPEED_RIGHT);
}
void go_Right()  //Turn right
{
  digitalWrite(RightDirectPin1, LOW);
  digitalWrite(RightDirectPin2, HIGH);
  digitalWrite(LeftDirectPin1, HIGH);
  digitalWrite(LeftDirectPin2, LOW);
  set_Motorspeed(SPEED_LEFT, 0);
}
void go_Back()  //Reverse
{
  digitalWrite(RightDirectPin1, LOW);
  digitalWrite(RightDirectPin2, HIGH);
  digitalWrite(LeftDirectPin1, LOW);
  digitalWrite(LeftDirectPin2, HIGH);
  set_Motorspeed(BACK_SPEED, BACK_SPEED);
}
void stop_Stop()  //Stop
{
  digitalWrite(RightDirectPin1, LOW);
  digitalWrite(RightDirectPin2, LOW);
  digitalWrite(LeftDirectPin1, LOW);
  digitalWrite(LeftDirectPin2, LOW);
  set_Motorspeed(0, 0);
}

void set_Motorspeed(int SPEED_L, int SPEED_R) {
  analogWrite(speedPinL, SPEED_L);
  analogWrite(speedPinR, SPEED_R);
}


//car motor control
void do_Drive_Tick() {

  if (Drive_Status == MANUAL_DRIVE) {
    switch (Drive_Num) {
      case GO_ADVANCE:
        go_Advance();
        Serial.println("go ahead");
        //  delay(AHEAD_TIME);
        break;
      case GO_LEFT:
        go_Left();
        //  delay(LEFT_TURN_TIME);
        Serial.println("TURN left");
        break;
      case GO_RIGHT:
        go_Right();
        //  delay(LEFT_TURN_TIME);
        Serial.println("TURN right");
        break;
      case GO_BACK:
        go_Back();
        // delay(BACK_TIME);
        Serial.println("GO back");
        break;
      case STOP_STOP:
        stop_Stop();
        //  JogTime = 0;
        Serial.println("STOP");
        break;
      default:
        break;
    }

  } else if (Drive_Status == AUTO_DRIVE_LF) {
    //Serial.println("auto track");
    // auto_tracking();
  }
}

void setup() {

  pinMode(RightDirectPin1, OUTPUT);
  pinMode(RightDirectPin2, OUTPUT);
  pinMode(speedPinL, OUTPUT);
  pinMode(LeftDirectPin1, OUTPUT);
  pinMode(LeftDirectPin2, OUTPUT);
  pinMode(speedPinR, OUTPUT);
  stop_Stop();  //stop move

  Serial.begin(9600);  // initialize serial for debugging

  Serial1.begin(115200);  // initialize serial for ESP module
  Serial1.print("AT+CIOBAUD=9600\r\n");
  Serial1.write("AT+RST\r\n");
  Serial1.begin(9600);  // initialize serial for ESP module

  WiFi.init(&Serial1);  // initialize ESP module

  // check for the presence of the shield
  if (WiFi.status() == WL_NO_SHIELD) {
    Serial.println("WiFi shield not present");
    while (true)
      ;  // don't continue
  }

  // Serial.print("Attempting to start AP ");
  //  Serial.println(ssid);
  //AP mode
  //status = WiFi.beginAP(ssid, 10, "", 0);

  //STA mode
  while (status != WL_CONNECTED) {
    Serial.print("Attempting to connect to WPA SSID: ");
    Serial.println(ssid);
    // Connect to WPA/WPA2 network
    status = WiFi.begin(ssid, pass);
  }
  Serial.println("You're connected to the network");
  Serial.println();

  printWifiStatus();

  Udp.begin(localPort);

  Serial.print("Listening on port ");
  Serial.println(localPort);
}

boolean flag = false;
void loop() {

  int packetSize = Udp.parsePacket();
  if (packetSize) {  // if you get a client,
    Serial.print("Received packet of size ");
    Serial.println(packetSize);
    int len = Udp.read(packetBuffer, 255);
    if (len > 0) {
      packetBuffer[len] = 0;
    }
    char c = packetBuffer[0];
    switch (c)  //serial control instructions
    {

      case 'A':
        Drive_Status = MANUAL_DRIVE;
        Drive_Num = GO_ADVANCE;
        WorkMode = "GO_ADVANCE";
        break;
      case 'L':
        Drive_Status = MANUAL_DRIVE;
        Drive_Num = GO_LEFT;
        WorkMode = "GO_LEFT";
        break;
      case 'R':
        Drive_Status = MANUAL_DRIVE;
        Drive_Num = GO_RIGHT;
        WorkMode = "GO_RIGHT";
        break;
      case 'B':
        Drive_Status = MANUAL_DRIVE;
        Drive_Num = GO_BACK;
        WorkMode = "GO_BACK";
        break;
      case 'E':
        Drive_Status = MANUAL_DRIVE;
        Drive_Num = STOP_STOP;
        WorkMode = "STOP_STOP";
        break;
      case 'T':
        Drive_Status = AUTO_DRIVE_LF;
        WorkMode = "line follow";
        break;

      default: break;
    }  //END OF ACTION SWITCH
  }
  do_Drive_Tick();
}  //end of loop

void printWifiStatus() {
  // print your WiFi shield's IP address
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);
}
/*
  // print where to go in the browser
  Serial.println();
  Serial.print("To see this page in action, connect to ");
  Seria;//
  */