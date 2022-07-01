#csv
#seikaku
import time
import ADS1256
import RPi.GPIO as GPIO
import csv,os

date = '20220629'
action = 'test'
round = '1'

GPIO.setmode(GPIO.BCM)
channel = 24 #p5 pin
GPIO.setup(channel,GPIO.OUT)

try:
    print("Program start\r\n")
    ADC = ADS1256.ADS1256()
    ADC.ADS1256_init()
    file_name = "{0}_{1}_{2}.csv".format(date,action,round)
    with open(os.path.join(os.getcwd(),"data",file_name),'w') as f:
        writer = csv.writer(f,lineterminator='\n')
        time.sleep(3)
        t1 = time.time()
        for i in range(500):
                GPIO.output(channel,HIGH)
                ADC_Value = ADC.ADS1256_GetAll()
                I1 = int(ADC_Value[0]*5000/0x7fffff)
                Q1 = int(ADC_Value[1]*5000/0x7fffff)
                I2 = int(ADC_Value[2]*5000/0x7fffff) 
                Q2 = int(ADC_Value[3]*5000/0x7fffff)
                time.sleep(0.01)
                GPIO.output(channel,LOW)
                ADC_Value = ADC.ADS1256_GetAll()
                I3 = -(int(ADC_Value[0]*5000/0x7fffff))
                Q3 = -(int(ADC_Value[1]*5000/0x7fffff))
                I4 = -(int(ADC_Value[2]*5000/0x7fffff))
                Q4 = -(int(ADC_Value[3]*5000/0x7fffff))
                writer.writerow([i,I1,Q1,I2,Q2,I3,Q3,I4,Q4])
                print (i,I1,Q1,I2,Q2,I3,Q3,I4,Q4)
                time.sleep(0.01)

except:
    GPIO.cleanup()
    print ("\r\nProgram end     ")
    exit()
t2 = time.time()
print(float(t2-t1))

