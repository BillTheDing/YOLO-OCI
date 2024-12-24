import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/ultralytics-main/ultralytics/cfg/models/v10/yolov10x.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/root/autodl-tmp/ultralytics-main/newdata3/data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=8,
                close_mosaic=0,
                workers=8,
                max_det=1000,
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='yolov10x',
                )