import os

MODEL_CACHE = "faceswap"

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    if not os.path.exists(MODEL_CACHE):
        os.makedirs(MODEL_CACHE)
    
     # Download inswapper_128.onnx
    os.system("cd /faceswap && wget http://43.153.104.112:814/inswapper_128.onnx.zip && unzip inswapper_128.onnx.zip")
    # Download GFPGANv1.4
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P ./faceswap")
   
    # Caches insight files
    os.system("mkdir -P /root/.insightface/models/buffalo_l/")
    os.system("wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && unzip buffalo_l.zip -d /root/.insightface/models/buffalo_l")

if __name__ == "__main__":
    download_model()