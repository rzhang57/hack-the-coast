import time
from dataParsing import readData
from engagementLogic import handleEngagement

def data_loop(state):
    while True:
        readData(state)
        handleEngagement(state)
        time.sleep(5)  #TODO communicate with sunny and decide the timing
                        # currently im settingn it to 6 because every 5 seconds
                        # there should be updated data
