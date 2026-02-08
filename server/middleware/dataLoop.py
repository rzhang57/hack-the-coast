import time
from dataProcessing import processData
from engagementLogic import handle_engagement

def data_loop(state):
    while True:
        raw = collect_raw_data()  # your camera/face/metrics
        processed = process_data(raw, state)
        handle_engagement(state, processed)
        time.sleep(3)  # or 10, or dynamic
