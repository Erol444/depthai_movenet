"""
This file is the template of the scripting node source code in edge mode
Substitution is made in MovenetDepthaiEdge.py
"""
import marshal

def pd_postprocess(inference):
    xnorm = []
    ynorm = []
    scores = []
    for i in range(17):
        xnorm.append(inference[3*i+1])
        ynorm.append(inference[3*i])
        scores.append(inference[3*i+2])
          
    # next_crop_region = determine_crop_region(scores, x, y) if ${_smart_crop} else init_crop_region
    return xnorm, ynorm, scores #, next_crop_region

node.warn("Processing node started")
# Defines the default crop region (pads the full image from both sides to make it a square image) 
# Used when the algorithm cannot reliably determine the crop region from the previous frame.
result_buffer = Buffer(498)
while True:
    # Receive movenet inference
    inference = node.io['from_pd_nn'].get().getLayerFp16("Identity")
    # Process it
    xnorm, ynorm, scores = pd_postprocess(inference)
    # Send result to the host
    result = {"xnorm":xnorm, "ynorm":ynorm, "scores":scores}
    result_serial = marshal.dumps(result)
    # Uncomment the following line to know the correct size of result_buffer
    # node.warn("result_buffer size: "+str(len(result_serial)))

    result_buffer.getData()[:] = result_serial
    node.io['to_host'].send(result_buffer)
