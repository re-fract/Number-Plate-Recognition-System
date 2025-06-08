from ultralytics import YOLO

# if __name__ == '__main__':
#     model = YOLO("yolov8n.pt")
#     model.train(
#         data="dataset/data.yaml",
#         epochs=100,
#         batch=24,
#         patience=10,
#         imgsz=640,
#     )

#
# if __name__ == '__main__':
#     model = YOLO("C:\\Users\\Sid\\PycharmProjects\\ANPR\\runs\\detect\\train\\weights\\last.pt")
#     model.train(
#         data="dataset/data.yaml",
#         epochs=100,
#         batch=24,
#         patience=25,
#         imgsz=640,
#         resume = True,
#     )

if __name__ == '__main__':
    model = YOLO("C:\\Users\\Sid\\PycharmProjects\\ANPR\\runs\\detect\\train\\weights\\best.pt")
    model.val(data='dataset\\data.yaml')
