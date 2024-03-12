# รายชื่อสมาชิกกลุ่ม
## นายถิรวัฒน์ พงศ์ปฏิสนธิ 6410450958
## นายธนภัทร เชื้อโตหลวง 6410401060  
## นายศรัณย์  วงษ์คำ 6410451415 

---
## คำอธิบายเกี่ยวกับไฟล์และโฟลเดอร์ที่ส่งมา
- ### Model
	- `Model ที่พร้อมใช้งาน`
- ### image_to_readme
	- `รูปภาพที่ใช้ใน readme.md`
- ### landmarks
	-  `โฟล์เดอร์ ไฟล์ที่ถูก preprocess แล้ว`
- ### videos
	-  `โฟล์เดอร์ วิดิโอต้นฉบับ`

---
## การดึงข้อมูล
```
ดึงข้อมูลจากเว็ปเข้าเครื่องผ่านเว็ปไซต์
- https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed/data
- https://paperswithcode.com/dataset/wlasl
```
---
## การ preprocess ข้อมูล
```
สำหรับแต่ละวิดีโอในชุดข้อมูลจะมีการตรวจตำแหน่งของมือ ใบหน้า และลำตัวด้วยจุดจำนวน 180 จุด โดยแบ่งตรวจจับแต่ละส่วนดังนี้
- บริเวณมือ 42 จุด (ข้างละ 21 จุด)
- ใบหน้า 132 จุด
- ลำตัว 6 จุด

สำหรับแต่ละเฟรมภายในวิดีโอ พิกัดของแต่ละจุดจะถูกเก็บเป็นตัวแปรอิสระของ 1 ตัวอย่าง
และกำหนดตัวแปรตามเป็นอาร์เรย์ที่มีความยาวเท่าจำนวนคำศัพท์ ทุกจำนวนภายในอาร์เรย์มีค่าเท่ากับ 0
ยกเว้นตำแหน่งที่มีหมายเลขสอดคล้องกับผลเฉลยของวิดีโอที่มีเฟรมนั้นประกอบอยู่
```
---
## การสร้างโมเดล
![model_Stucture](/image_to_readme/model_structure.png)
---
## การโหลดข้อมูล
```
co = {}
try:
    for i in tqdm(range(len(data)), ncols=100):
        gloss = data[i]['gloss']
        if gloss not in co:
            co[gloss] = 1
        else:
            co[gloss] += 1
        
        npy_path = os.path.join(npy_dir, f"{gloss}{co[gloss]}.npy")
        if os.path.exists(npy_path):
            continue
        
        video_path = data[i]['video_path']
        start = data[i]['frame_start']
        end = data[i]['frame_end']
        
        try:
            video_landmarks = get_video_landmarks(video_path, start, end)
            np.save(npy_path, video_landmarks)
            
        except Exception as e:
            print(f"\nError encoding {video_path}\n{e}")
            continue
        
        clear_output(wait=True)

except KeyboardInterrupt:
    print("\nLoading process interrupted by user.")

```
---
## การเทรนโมเดล
```
initial_learning_rate = 0.001
lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=10000, decay_rate=0.9)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',  # Metric to monitor for early stopping
    mode='min',  # Set mode to 'min' for minimizing the metric
    patience=15,  # Number of epochs with no improvement before stopping
    restore_best_weights=True,  # Restore the best model weights
    verbose=1
)
model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])
model_training_history = model.fit(X_train, y_train, batch_size=32, epochs=50 , callbacks=[early_stopping])
```
---
## การวัดประสิทธิภาพโมเดล

![img_ef](/image_to_readme/Img_ef.png)

---

## การใช้งานโมเดล

![toto_1](/image_to_readme/toto_1.png)

![toto_2](/image_to_readme//toto_2.png)

![Ozone_3](/image_to_readme/Ozone_3.png)
