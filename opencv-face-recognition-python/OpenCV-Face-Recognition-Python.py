import cv2
import os
import numpy as np

subjects = ["", "Quoc Thinh", "Tuyet Mai"]
# chức năng phát hiện khuôn mặt bằng OpenCV
def detect_face(img):
    #chuyển đổi hình ảnh đầu vào thành hình ảnh màu xám như máy dò khuôn mặt opencv mong đợi hình ảnh màu xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load Máy dò tìm OpenCV there cũng là một phân loại Haar chính xác hơn nhưng chậm
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if không có khuôn mặt nào được phát hiện sau đó trả về img gốc
    if (len(faces) == 0):
        return None, None
    
    # theo giả định rằng sẽ chỉ có một khuôn mặt,
    # giải nén vùng mặt
    (x, y, w, h) = faces[0]
    
    # chỉ trả lại phần khuôn mặt của hình ảnh
    return gray[y:y+w, x:x+h], faces[0]


#hàm này sẽ đọc hình ảnh đào tạo của mọi người, phát hiện khuôn mặt từ mỗi hình ảnh
# và sẽ trả về hai danh sách có cùng kích thước, một danh sách
# khuôn mặt và một danh sách nhãn khác cho mỗi khuôn mặt
def prepare_training_data(data_folder_path):
    
    #------BƯỚC 1--------
     # lấy các thư mục (một thư mục cho mỗi đối tượng) trong thư mục dữ liệu
    dirs = os.listdir(data_folder_path)
    
    #danh sách để giữ tất cả các khuôn mặt
    faces = []
    #danh sách để giữ nhãn cho tất cả các đối tượng
    labels = []
    
    for dir_name in dirs:
        
        #thư mục của em bắt đầu bằng chữ 's' vì vậy
        if not dir_name.startswith("s"):
            continue;
            
        #------BƯỚC-2--------
        # trích xuất số nhãn của chủ đề từ dir_name
        #định dạng của tên dir = s+label
        #, vì vậy việc xóa chữ 's' khỏi dir_name sẽ cung cấp cho em nhãn
        label = int(dir_name.replace("s", ""))
        
        # xây dựng đường dẫn của thư mục chứa hình ảnh cho chủ đề hiện 
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #lấy tên hình ảnh nằm trong thư mục chủ đề đã cho
        subject_images_names = os.listdir(subject_dir_path)
        
        #------BƯỚC 3--------
        # đi qua từng tên hình ảnh, đọc hình ảnh,
        # phát hiện khuôn mặt và thêm khuôn mặt vào danh sách khuôn mặt
        for image_name in subject_images_names:
            
            # bỏ qua các tệp hệ thống như .DS_Store
            if image_name.startswith("."):
                continue;
            
            #đường dẫn hình ảnh
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            
            #phát hiện khuôn mặt
            face, rect = detect_face(image)
            
            #------STEP-4--------
            #bỏ qua các khuôn mặt không được phát hiện.và chỉ lấy các khuông mặt được nhận diện
            if face is not None:
                #thêm khuông mặt vào danh sách faces
                faces.append(face)
                #thêm nhãn cho khuông mặt này
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels

print("Preparing data...")
faces, labels = prepare_training_data("training-data")

#create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#đào tạo nhận dạng khuôn mặt
face_recognizer.train(faces, np.array(labels))

#Hàm vẽ hình chữ nhật trên hình ảnh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    #cv2.rectangle(img, topLeftPoint, bottomRightPoint, rgbColor, lineWidth)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#hàm viết tên lên ảnh 
def draw_text(img, text, x, y):
    #cv2.putText(img, text, startPoint, font, fontSize, rgbColor, lineWidth)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

#hàm phát hiện khuông mặt và vẽ hình chữ nhật lên khuông mặt và ghi tên của đối tượng lên hình
def predict(test_img):
    img = test_img.copy()
    #phát hiện khuôn mặt từ hình ảnh
    face, rect = detect_face(img)

    #dự đoán hình ảnh bằng cách sử dụng nhận dạng khuôn mặt
    label, confidence = face_recognizer.predict(face)
    #lấy tên của nhãn tương ứng được trả về bởi nhận dạng khuôn mặt
    label_text = subjects[label]
    
    #vẽ một hình chữ nhật xung quanh khuôn mặt được phát hiện
    draw_rectangle(img, rect)
    #vẽ tên của người dự đoán
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img

print("Predicting images...")
#load hình ảnh kiểm tra
test_img = cv2.imread("test-data/thinh.jpg")
#thực hiện dự đoán
predicted_img = predict(test_img)

print("Prediction complete")
cv2.imshow("Image predicted", cv2.resize(predicted_img, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
