import json
import os

_RECAPTION_DIR = os.path.dirname(__file__)
json1 = os.path.join(_RECAPTION_DIR, 'avsync-test-72B-captions.json')
json2 = os.path.join(_RECAPTION_DIR, 'avsync-test-0129.json')

with open(json1, 'r') as f:
    data1 = json.load(f)
with open(json2, 'r') as f:
    data2 = json.load(f)

for key in data1:
    if key in data2:
        data1[key]['audio_video_caption'] = data2[key].get('audio_video_caption', '')

with open(json2, 'w') as f:
    json.dump(data1, f, indent=4, ensure_ascii=False)
