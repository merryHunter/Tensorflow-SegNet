# How to run from scratch

1. Download https://github.com/alexgkendall/SegNet-Tutorial to the same folder as the Tensorflow-SegNet
2. Rename it to SegNet
3. Dependencies:  
   `virtualenv env `  
   `source env/bin/activate`  
   `pip install tensorflow-gpu==1.3.0` (originally you need 1.0, but seems it works with the latest version)  
   `pip install scikit-image`  
4. **Train**:  
   `python main.py --image_dir=../SegNet/CamVid/train.txt --val_dir=../SegNet/CamVid/val.txt --log_dir=../logs/ --batch_size=5 `  
   **Finetune from ckpt**:  
   `python main.py --finetune=../logs/model.ckpt-#### --image_dir=../SegNet/CamVid/train.txt --val_dir=../SegNet/CamVid/val.txt --log_dir=../logs/ --batch_size=5`  
   **Test**:  
    `python main.py --testing=../logs/model.ckpt-#### --log_dir=../logs/ --test_dir=../SegNet/CamVid/test.txt --batch_size=5 --save_image=True`
    
    
# How to convert KITTI to TFRecord  

`python tfrecords_converter.py --kitti_path /full_path_kitti_data_road --output output_dir/`