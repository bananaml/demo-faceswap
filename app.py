from potassium import Potassium, Request, Response
import cv2
import gfpgan
import base64
import numpy as np
import insightface
import onnxruntime
from insightface.app import FaceAnalysis

app = Potassium("faceswap")

@app.init
def init():
    face_swapper = insightface.model_zoo.get_model(
        'faceswap/inswapper_128.onnx',
        providers=onnxruntime.get_available_providers()
    )
    face_enhancer = gfpgan.GFPGANer(
        model_path='faceswap/GFPGANv1.4.pth',
        upscale=1
    )
    face_analyser = FaceAnalysis(name='buffalo_l')
    face_analyser.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )
    context = {
        "swapper": face_swapper,
        "enhancer": face_enhancer,
        "analyser": face_analyser
    }
    return context

def get_face(face_analyser, img_data):
    analysed = face_analyser.get(img_data)
    try:
        largest = max(analysed, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return largest
    except:
        print("No face found") 
        return None
        
# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    swapper = context.get("swapper")
    enhancer = context.get("enhancer")
    analyser = context.get("analyser")
    target_image = request.json.get("target_image")
    image_decoded = base64.b64decode(target_image)
    image_np_array = np.frombuffer(image_decoded, np.uint8)
    frame = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
    swap_image = request.json.get("swap_image")
    image_decoded = base64.b64decode(swap_image)
    image_np_array = np.frombuffer(image_decoded, np.uint8)
    swap_image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
    face = get_face(analyser, frame)
    source_face = get_face(analyser, swap_image)
    try:
        print(frame.shape, face.shape, source_face.shape)
    except:
        print("printing shapes failed.")
    result = swapper.get(frame, face, source_face, paste_back=True)
    _, _, result = enhancer.enhance(
        result,
        paste_back=True
    )
    out_path = "out.png"
    cv2.imwrite(out_path, result)
    with open(out_path, 'rb') as file:
        image_bytes = file.read()
    output_bytes = base64.b64encode(image_bytes)
    output = output_bytes.decode('utf-8')

    return Response(
        json = {"output": output}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()